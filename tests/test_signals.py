import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError, connection, transaction
from django.db.models import QuerySet
from django.db.models.signals import post_delete, pre_delete
from django.test.utils import CaptureQueriesContext
from pytest_django import (
    DjangoAssertNumQueries,
    DjangoCaptureOnCommitCallbacks,
    Settings,
)

from django_model_rag import BaseExtractor, NormalizedDocument, rag
from tests.recording import (
    FAILING_ON_KEY_BACKEND,
    TRACKED_BACKEND,
    FailingReplaceError,
    TrackedRecordingOutput,
)
from tests.testapp.models import (
    Album,
    Banner,
    Bin,
    BonusTrack,
    Bookmark,
    Category,
    CategoryProxy,
    Citation,
    ClearanceProduct,
    Course,
    Craftsman,
    Depot,
    Excerpt,
    Exhibit,
    FeaturedProduct,
    Guild,
    Lesson,
    Meetup,
    Note,
    Notice,
    Offer,
    Page,
    PageIntro,
    Photo,
    Product,
    Remark,
    Review,
    Seminar,
    Session,
    Shelf,
    Showroom,
    Spotlight,
    Supplier,
    SupplierProfile,
    Tag,
    Talk,
    TextPlugin,
    TextPluginProxy,
    Theme,
    Topic,
    TopicProxy,
    Venue,
    Warehouse,
    Workshop,
)

# The logger the package reports a failed commit callback on.
PACKAGE_LOGGER = "django_model_rag"


def _replaced(
    built_outputs: list[TrackedRecordingOutput],
) -> list[Mapping[str, Sequence[NormalizedDocument]]]:
    """Every replace call received, across every output built, in call order."""
    return [groups for output in built_outputs for groups in output.replaced]


def _received_groups(
    built_outputs: list[TrackedRecordingOutput],
) -> dict[str, list[NormalizedDocument]]:
    """Every group received, across every output built, merged into one mapping."""
    return {
        source_key: list(group)
        for groups in _replaced(built_outputs)
        for source_key, group in groups.items()
    }


def _package_log_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The records captured from the package's logger, and only those."""
    return [record for record in caplog.records if record.name == PACKAGE_LOGGER]


def _statements(queries: CaptureQueriesContext) -> list[str]:
    """The kind of each captured query (its first SQL keyword), in query order."""
    return [query["sql"].split()[0] for query in queries.captured_queries]


def _register_categories_by_name() -> None:
    """Register Category, each instance extracted to one document: its name."""

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)


def _register_products_by_name() -> None:
    """Register Product, each instance extracted to one document: its name."""

    @rag.register_extractor(Product)
    class ProductExtractor(BaseExtractor[Product]):
        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)


def _register_pages_following_their_plugins() -> None:
    """Register only Page, following its text plugins: TextPlugin itself is not."""
    rag.register(Page, follow=["text_plugins"])


def _register_plugins_by_their_page_title() -> None:
    """Register only TextPlugin, with a custom extractor reading its page's title
    before its body: depends_on names the page, whose saves change its
    documents. Page itself is not registered."""

    @rag.register_extractor(TextPlugin, depends_on=["page"])
    class TextPluginExtractor(BaseExtractor[TextPlugin]):
        def extract(self, instance: TextPlugin) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.page.title}: {instance.body}"
            )


def _register_venues_following_their_seminars() -> None:
    """Register only Venue, by its name, following its seminars by the reverse of
    the multi-column ForeignObject ``venue``: Seminar itself is not."""
    rag.register(Venue, fields=["name"], follow=["seminars"])


def _create_the_hall_and_its_acoustics_seminar() -> tuple[Venue, Seminar]:
    """Create the Halle Tony Garnier, a venue of Lyon, and its Acoustics seminar."""
    hall = Venue.objects.create(city="Lyon", name="Halle Tony Garnier")
    acoustics = Seminar.objects.create(
        title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )
    return hall, acoustics


def _create_the_transbordeur_and_its_seminar() -> Seminar:
    """Create the Transbordeur, a venue of Lyon, and its Stage lighting seminar."""
    Venue.objects.create(city="Lyon", name="Transbordeur")
    return Seminar.objects.create(
        title="Stage lighting", venue_city="Lyon", venue_name="Transbordeur"
    )


def _create_a_desk_lamp(category: Category) -> FeaturedProduct:
    """Create a Desk lamp, a FeaturedProduct: a Product row and its child row."""
    return FeaturedProduct.objects.create(
        name="Desk lamp",
        description="A lamp for the desk.",
        price="25.00",
        category=category,
        tagline="Light up your work",
    )


def _create_a_plain_desk_lamp(category: Category) -> Product:
    """Create a Desk lamp, a plain Product: a single row."""
    return Product.objects.create(
        name="Desk lamp",
        description="A lamp for the desk.",
        price="25.00",
        category=category,
    )


def _create_a_bulb(category: Category) -> Product:
    """Create a Bulb, a plain Product: a single row."""
    return Product.objects.create(
        name="Bulb",
        description="A bulb for the lamp.",
        price="5.00",
        category=category,
    )


def _create_a_hammer(category: Category) -> Product:
    """Create a Hammer, a plain Product: a single row."""
    return Product.objects.create(
        name="Hammer",
        description="A hammer for nails.",
        price="15.00",
        category=category,
    )


def _create_the_woodworking_topic() -> Topic:
    """Create the Woodworking topic."""
    return Topic.objects.create(
        summary="Joints and finishes.", title="Woodworking", slug="woodworking"
    )


@pytest.mark.django_db
def test_saving_a_registered_instance_replaces_its_group_once_its_transaction_commits(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        # A rolled-back transaction must leave the output untouched: nothing
        # may reach it before the commit.
        assert _replaced(built_outputs) == []

    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{lighting.pk}": [
                NormalizedDocument(
                    text="Lighting",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=lighting.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_through_a_proxy_replaces_the_group_of_the_registered_model(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the concrete model is registered, not its proxy.
    _register_categories_by_name()

    # Django sends post_save with the proxy as its sender, not Category.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = CategoryProxy.objects.create(name="Lighting")
        assert _replaced(built_outputs) == []

    # The group is the registered model's, under its label, not the proxy's.
    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{lighting.pk}": [
                NormalizedDocument(
                    text="Lighting",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=lighting.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_multi_table_child_replaces_the_group_of_its_registered_parent(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the parent is registered, not its multi-table child.
    _register_products_by_name()

    # Created in a commit of its own, unregistered: only the child's save
    # below is observed.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    built_outputs.clear()

    # Django sends post_save once, with the child as its sender, not Product.
    with django_capture_on_commit_callbacks(execute=True):
        lamp = _create_a_desk_lamp(lighting)
        assert _replaced(built_outputs) == []

    # The group is the parent row's, under the parent's label, and its
    # document is extracted from a Product, not from the child.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_child_with_a_primary_key_of_its_own_replaces_its_parents_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the parent is registered, not its multi-table child.
    _register_products_by_name()

    # Created in a commit of its own, unregistered: only the child's save
    # below is observed.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    built_outputs.clear()

    # The child's primary key is its code, not the Product's: the parent row
    # is reached by the explicit parent link, ``product``.
    with django_capture_on_commit_callbacks(execute=True):
        lamp = ClearanceProduct.objects.create(
            code="CLR-1",
            name="Desk lamp",
            description="A lamp for the desk.",
            price="25.00",
            category=lighting,
        )
        assert _replaced(built_outputs) == []

    # The group is the parent row's, under the parent's label and the parent's
    # primary key, not the child's code, and its document is extracted from
    # the Product row.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.product_id}": [
                NormalizedDocument(
                    text="Desk lamp",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.product_id,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_registered_multi_table_child_also_replaces_its_parents_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Both the parent and its multi-table child are registered, each with an
    # extractor of its own, whose texts tell their documents apart.
    @rag.register_extractor(Product)
    class ProductExtractor(BaseExtractor[Product]):
        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(instance, text=f"Product: {instance.name}")

    @rag.register_extractor(FeaturedProduct)
    class FeaturedProductExtractor(BaseExtractor[FeaturedProduct]):
        def extract(self, instance: FeaturedProduct) -> NormalizedDocument:
            return self.build_document(instance, text=f"Featured: {instance.tagline}")

    # Created in a commit of its own, unregistered: only the child's save
    # below is observed.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    built_outputs.clear()

    # Django sends post_save once, with the child as its sender, not Product.
    with django_capture_on_commit_callbacks(execute=True):
        lamp = _create_a_desk_lamp(lighting)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per label is not what this test is about.
    received = _received_groups(built_outputs)
    # Each group under its own label, from its own extractor: the child's
    # document for the child, the parent's document for the parent row.
    assert received == {
        f"testapp.featuredproduct:{lamp.pk}": [
            NormalizedDocument(
                text="Featured: Light up your work",
                source_app_label="testapp",
                source_model="featuredproduct",
                source_pk=lamp.pk,
            ),
        ],
        f"testapp.product:{lamp.pk}": [
            NormalizedDocument(
                text="Product: Desk lamp",
                source_app_label="testapp",
                source_model="product",
                source_pk=lamp.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_followed_related_instance_replaces_the_group_that_follows_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callback of the
    # Page's own save never runs, so only the plugin's save below is observed.
    page = Page.objects.create(title="About us", slug="about-us")

    with django_capture_on_commit_callbacks(execute=True):
        TextPlugin.objects.create(page=page, body="We build chairs by hand.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Page's group, with the plugin's text after the Page's own title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.page:{page.pk}": [
                NormalizedDocument(
                    text="About us\n\nWe build chairs by hand.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=page.pk,
                    title="About us",
                    url="/pages/about-us/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_followed_by_foreign_key_replaces_the_group_that_follows_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Product's group, with the category's new name after the Product's
    # own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_read_through_a_lookup_path_replaces_the_group_reading_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, reading its category's name through a
    # lookup path, with no follow: Category itself is not registered.
    rag.register(Product, fields=["name", "category__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the Product, as follow=["category"] does: its
    # group, with the category's new name after the Product's own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_notice_read_as_language_through_a_lookup_path_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Excerpt is registered, reading its language from its notice
    # through a lookup path, with no follow: Notice itself is not registered.
    rag.register(Excerpt, fields=["title"], language_field="notice__language")

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the notice's save below is observed.
    notice = Notice.objects.create(title="Avis", language="fr")
    excerpt = Excerpt.objects.create(title="Bonjour", notice=notice)

    with django_capture_on_commit_callbacks(execute=True):
        notice.language = "en"
        notice.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the Excerpt: its group, with the notice's new
    # language.
    assert _replaced(built_outputs) == [
        {
            f"testapp.excerpt:{excerpt.pk}": [
                NormalizedDocument(
                    text="Bonjour",
                    source_app_label="testapp",
                    source_model="excerpt",
                    source_pk=excerpt.pk,
                    title="Bonjour",
                    language="en",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_bookmark_read_as_url_through_a_lookup_path_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Citation is registered, reading its url from its bookmark
    # through a lookup path, with no follow: Bookmark itself is not registered.
    rag.register(Citation, fields=["title"], url_field="bookmark__link")

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the bookmark's save below is observed.
    bookmark = Bookmark.objects.create(title="Docs", link="/docs/a/")
    citation = Citation.objects.create(title="Linked", bookmark=bookmark)

    with django_capture_on_commit_callbacks(execute=True):
        bookmark.link = "/docs/b/"
        bookmark.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the Citation: its group, with the bookmark's new
    # link as its url.
    assert _replaced(built_outputs) == [
        {
            f"testapp.citation:{citation.pk}": [
                NormalizedDocument(
                    text="Linked",
                    source_app_label="testapp",
                    source_model="citation",
                    source_pk=citation.pk,
                    title="Linked",
                    url="/docs/b/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_read_as_title_through_a_lookup_path_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, reading its title from its category
    # through a lookup path, with no follow and no other path through the
    # category: Category itself is not registered.
    rag.register(Product, fields=["name"], title_field="category__name")

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the Product: its group, with the category's new
    # name as its title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Lamps",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_read_two_foreign_keys_deep_replaces_the_group_reading_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: neither
    # Product nor Category is registered, so the category reaches the offer
    # only through the path.
    rag.register(Offer, fields=["title", "product__category__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_plain_desk_lamp(lighting)
    rake = Product.objects.create(
        name="Rake",
        description="Wooden.",
        price="14.50",
        category=garden,
    )
    spring = Offer.objects.create(title="Spring sale", product=lamp)
    Offer.objects.create(title="Autumn sale", product=rake)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the offer of a product in the category saved, the
    # last link of the path: its group, with the category's new name after the
    # offer's own title. The offer of a product in another category is not
    # sent.
    assert _replaced(built_outputs) == [
        {
            f"testapp.offer:{spring.pk}": [
                NormalizedDocument(
                    text="Spring sale\n\nLamps",
                    source_app_label="testapp",
                    source_model="offer",
                    source_pk=spring.pk,
                    title="Spring sale",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_product_in_the_middle_of_a_lookup_path_replaces_the_group_reading_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: neither
    # Product nor Category is registered, so the product reaches the offer
    # only through the path.
    rag.register(Offer, fields=["title", "product__category__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the product's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_plain_desk_lamp(lighting)
    rake = Product.objects.create(
        name="Rake",
        description="Wooden.",
        price="14.50",
        category=garden,
    )
    spring = Offer.objects.create(title="Spring sale", product=lamp)
    Offer.objects.create(title="Autumn sale", product=rake)

    with django_capture_on_commit_callbacks(execute=True):
        lamp.category = garden
        lamp.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the offer of the product saved, the middle link
    # of the path: its group, with the name of the product's new category
    # after the offer's own title. The offer of another product is not sent.
    assert _replaced(built_outputs) == [
        {
            f"testapp.offer:{spring.pk}": [
                NormalizedDocument(
                    text="Spring sale\n\nGarden",
                    source_app_label="testapp",
                    source_model="offer",
                    source_pk=spring.pk,
                    title="Spring sale",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_both_followed_and_read_by_a_path_replaces_each_group_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, reaching its category twice: through
    # follow=["category"] and through the lookup path "category__name".
    # Category itself is not registered.
    rag.register(Product, fields=["name", "category__name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)
    bulb = _create_a_bulb(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The two declarations reaching the same category resync each product once:
    # one replace call holding each product's group, not one call per
    # declaration. Each text holds the category's new name twice, once read
    # through the path and once through the followed relation.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
            f"testapp.product:{bulb.pk}": [
                NormalizedDocument(
                    text="Bulb\n\nLamps\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=bulb.pk,
                    title="Bulb",
                    url=f"/products/{bulb.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db(transaction=True)
def test_a_category_read_two_links_deep_saved_with_no_output_writes_no_row() -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: neither
    # Product nor Category is registered, so the category reaches the offer
    # only through the path.
    rag.register(Offer, fields=["title", "product__category__name"])

    # No transaction around the save (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the save must fail before its
    # INSERT does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        Category.objects.create(name="Lighting")

    assert not Category.objects.exists()


@pytest.mark.django_db
def test_saving_a_category_followed_by_two_products_replaces_both_in_one_batch(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)
    bulb = _create_a_bulb(lighting)
    # A product of another category: the save below does not change its group.
    seating = Category.objects.create(name="Seating")
    Product.objects.create(
        name="Chair",
        description="A chair to sit on.",
        price="80.00",
        category=seating,
    )

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # One replace call holding the groups of both followers, as one batch of
    # SyncPipeline.run_queryset() does, and no group of the other category's
    # product.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
            f"testapp.product:{bulb.pk}": [
                NormalizedDocument(
                    text="Bulb\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=bulb.pk,
                    title="Bulb",
                    url=f"/products/{bulb.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_followed_by_more_rows_than_sqlite_variables_replaces_all(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    sqlite_variable_limit_lowered: int,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # More followers than SQLite accepts variables in one query. Created
    # outside the captured callbacks: bulk_create sends no signal anyway, so
    # only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamps = Product.objects.bulk_create(
        Product(
            name=f"Lamp {number}",
            description="A lamp.",
            price="25.00",
            category=lighting,
        )
        for number in range(sqlite_variable_limit_lowered + 100)
    )

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()

    # Every follower's group, with the category's new name after the
    # Product's own name: none is left out for the size of the query.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{lamp.pk}": [
            NormalizedDocument(
                text=f"{lamp.name}\n\nLamps",
                source_app_label="testapp",
                source_model="product",
                source_pk=lamp.pk,
                title=lamp.name,
                url=f"/products/{lamp.pk}/",
            ),
        ]
        for lamp in lamps
    }


@pytest.mark.django_db
def test_creating_a_category_followed_by_foreign_key_reads_nothing_and_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # A working output, so that only the new row can keep the save from
    # sending anything.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits: neither the save nor the commit may read.
    with (
        django_assert_num_queries(1) as queries,
        django_capture_on_commit_callbacks(execute=True) as callbacks,
    ):
        # A row just created: no Product can point to it yet.
        Category.objects.create(name="Lighting")

    # The only query is the save's INSERT: no lookup of followers. Nothing is
    # deferred to the commit, and no output is built.
    statements = _statements(queries)
    assert statements == ["INSERT"]
    assert callbacks == []
    assert built_outputs == []


@pytest.mark.django_db
def test_saving_a_category_followed_by_foreign_key_looks_its_followers_up_once(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    lighting = Category.objects.create(name="Lighting")
    _create_a_plain_desk_lamp(lighting)

    # The commit callbacks are not run: only the save itself is counted.
    with (
        django_capture_on_commit_callbacks(execute=False),
        django_assert_num_queries(2) as queries,
    ):
        lighting.name = "Lamps"
        lighting.save()

    # The save's UPDATE and one lookup of the products following the
    # category: the lookup before the save takes the place of the one after
    # it, and the category's row as committed is not loaded for a link to its
    # primary key.
    statements = _statements(queries)
    assert sorted(statements) == ["SELECT", "UPDATE"]
    lookup = queries.captured_queries[statements.index("SELECT")]["sql"]
    assert 'FROM "testapp_product"' in lookup


@pytest.mark.django_db
def test_saving_through_a_proxy_of_a_category_followed_by_foreign_key_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: neither Category nor its proxy is.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)

    # Django sends post_save with the proxy as its sender, not Category.
    with django_capture_on_commit_callbacks(execute=True):
        lamps = CategoryProxy.objects.get(pk=lighting.pk)
        lamps.name = "Lamps"
        lamps.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Product's group, with the category's new name after the Product's
    # own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{lamp.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_category_followed_by_a_foreign_key_to_its_proxy_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Banner is registered, following its category through its own
    # foreign key, whose model is the proxy CategoryProxy: neither Category nor
    # its proxy is.
    rag.register(Banner, fields=["title"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    banner = Banner.objects.create(title="Spring sale", category_id=lighting.pk)

    # A plain Category, not its proxy: Django sends post_save with Category as
    # its sender, while the Banner's foreign key names the proxy.
    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Banner's group, with the category's new name after the Banner's own
    # title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.banner:{banner.pk}": [
                NormalizedDocument(
                    text="Spring sale\n\nLamps",
                    source_app_label="testapp",
                    source_model="banner",
                    source_pk=banner.pk,
                    title="Spring sale",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_multi_table_child_of_a_product_followed_by_foreign_key_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Review is registered, following its product through its own
    # foreign key: neither Product nor FeaturedProduct is.
    rag.register(Review, follow=["product"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the featured product's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_desk_lamp(lighting)
    review = Review.objects.create(title="Sturdy", product=lamp)

    # Django sends post_save with FeaturedProduct as its sender, not Product,
    # though the save writes the Product row the Review follows.
    with django_capture_on_commit_callbacks(execute=True):
        lamp.name = "Floor lamp"
        lamp.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Review's group, with the product's new text after the Review's own
    # title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.review:{review.pk}": [
                NormalizedDocument(
                    text="Sturdy\n\nFloor lamp\n\nA lamp for the desk.\n\nNew",
                    source_app_label="testapp",
                    source_model="review",
                    source_pk=review.pk,
                    title="Sturdy",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_warehouse_followed_by_a_foreign_key_to_its_code_replaces_its_shelves(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Shelf is registered, following its warehouse through its own
    # foreign key, which holds the Warehouse's code, not its primary key:
    # Warehouse itself is not.
    rag.register(Shelf, follow=["warehouse"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the warehouse's save below is observed.
    north = Warehouse.objects.create(name="North depot", code="north")
    timber = Shelf.objects.create(warehouse=north, label="Timber")
    # Another warehouse whose code is the North depot's primary key as text: a
    # shelf matched by comparing that primary key with the stored code would
    # be this one's, not the North depot's.
    south = Warehouse.objects.create(name="South depot", code=str(north.pk))
    Shelf.objects.create(warehouse=south, label="Paint")

    with django_capture_on_commit_callbacks(execute=True):
        north.name = "North hall"
        north.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Only the North depot's shelf's group, with the warehouse's new name after
    # the shelf's own label, and no group of the other warehouse's shelf.
    assert _replaced(built_outputs) == [
        {
            f"testapp.shelf:{timber.pk}": [
                NormalizedDocument(
                    text="Timber\n\nNorth hall",
                    source_app_label="testapp",
                    source_model="shelf",
                    source_pk=timber.pk,
                    title="Timber",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_changing_the_code_of_a_warehouse_followed_by_it_replaces_the_shelves_naming_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Shelf is registered, following its warehouse through its own
    # foreign key, which holds the Warehouse's code, not its primary key:
    # Warehouse itself is not.
    rag.register(Shelf, follow=["warehouse"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the warehouse's code change below is observed.
    north = Warehouse.objects.create(name="North depot", code="north")
    timber = Shelf.objects.create(warehouse=north, label="Timber")
    # Another warehouse and its shelf, untouched by the code change.
    south = Warehouse.objects.create(name="South depot", code="south")
    Shelf.objects.create(warehouse=south, label="Paint")

    with django_capture_on_commit_callbacks(execute=True):
        # The code is the column the shelves point to the warehouse by: at the
        # save, the North depot's shelf still holds the old code, so only the
        # warehouse as it was before the save tells which shelves named it.
        north.code = "north-hall"
        north.save()
        # The shelves are then moved to the new code, in the same transaction,
        # by an update that sends no signal: the foreign key constraint is
        # checked at the commit, and the shelf's own save is not observed.
        Shelf.objects.filter(warehouse_id="north").update(warehouse_id="north-hall")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of the shelf that named the North depot, as committed, with
    # the warehouse's name after the shelf's own label; no group of the other
    # warehouse's shelf.
    assert _replaced(built_outputs) == [
        {
            f"testapp.shelf:{timber.pk}": [
                NormalizedDocument(
                    text="Timber\n\nNorth depot",
                    source_app_label="testapp",
                    source_model="shelf",
                    source_pk=timber.pk,
                    title="Timber",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_depot_with_a_null_code_replaces_no_group_of_the_bins_with_no_depot(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Bin is registered, following its depot through its own nullable
    # foreign key, which holds the Depot's code, not its primary key: Depot
    # itself is not.
    rag.register(Bin, follow=["depot"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the depot's save below is observed. The depot's
    # code is null.
    unnamed = Depot.objects.create(name="Unnamed depot", code=None)
    # Its nullable foreign key is null: it points to no Depot, not even to the
    # one whose code is null, as the database never joins NULL to NULL.
    Bin.objects.create(depot=None, label="Spare parts")

    with django_capture_on_commit_callbacks(execute=True):
        unnamed.name = "Overflow depot"
        unnamed.save()

    # No bin follows the depot: no group is replaced.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_saving_a_page_followed_by_a_forward_one_to_one_replaces_the_group_of_its_intro(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the PageIntro is registered, following its page through its own
    # one-to-one field: Page itself is not.
    rag.register(PageIntro, follow=["page"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the page's save below is observed.
    about = Page.objects.create(title="About us", slug="about-us")
    intro = PageIntro.objects.create(page=about, body="We build chairs by hand.")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The intro's group, with the page's new title after the intro's own body.
    assert _replaced(built_outputs) == [
        {
            f"testapp.pageintro:{intro.pk}": [
                NormalizedDocument(
                    text="We build chairs by hand.\n\nOur workshop",
                    source_app_label="testapp",
                    source_model="pageintro",
                    source_pk=intro.pk,
                    title="We build chairs by hand.",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_page_a_custom_extractor_depends_on_replaces_the_groups_of_its_plugins(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_plugins_by_their_page_title()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the page's save below is observed.
    about = Page.objects.create(title="About us", slug="about-us")
    chairs = TextPlugin.objects.create(page=about, body="We build chairs by hand.")
    tables = TextPlugin.objects.create(page=about, body="And tables too.")
    # A plugin of another page: the save below does not change its group.
    contact = Page.objects.create(title="Contact", slug="contact")
    TextPlugin.objects.create(page=contact, body="Write to us.")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The groups of the saved page's plugins, with the page's
    # new title, and no group of the other page's plugin.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="Our workshop: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
            ),
        ],
        f"testapp.textplugin:{tables.pk}": [
            NormalizedDocument(
                text="Our workshop: And tables too.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=tables.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_a_plugin_bulk_created_after_its_page_save_is_replaced_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_plugins_by_their_page_title()

    # Created outside the captured callbacks: its commit callbacks never run,
    # and it has no plugin yet.
    about = Page.objects.create(title="About us", slug="about-us")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Attached after the page's save, in the same transaction, by a write
        # that sends no signal: only the page's save is observed.
        (chairs,) = TextPlugin.objects.bulk_create(
            [TextPlugin(page=about, body="We build chairs by hand.")]
        )

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The plugin attached after the save is a follower of the
    # page at the commit, so its group is replaced, with the page's new title.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="Our workshop: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_a_plugin_moved_by_update_after_its_page_save_is_replaced_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_plugins_by_their_page_title()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The plugin starts on another page; the page it moves to has none.
    news = Page.objects.create(title="News", slug="news")
    chairs = TextPlugin.objects.create(page=news, body="We build chairs by hand.")
    about = Page.objects.create(title="About us", slug="about-us")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Moved to the page after the page's save, in the same transaction, by
        # a write that sends no signal: only the page's save is observed.
        TextPlugin.objects.filter(pk=chairs.pk).update(page=about)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The plugin moved after the save is a follower of the page
    # at the commit, so its group is replaced, with the page's new title.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="Our workshop: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_a_plugin_moved_away_by_update_after_its_page_save_is_replaced_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_plugins_by_their_page_title()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The plugin starts on the saved page; the page it moves to has none.
    about = Page.objects.create(title="About us", slug="about-us")
    chairs = TextPlugin.objects.create(page=about, body="We build chairs by hand.")
    news = Page.objects.create(title="News", slug="news")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Moved away from the page after the page's save, in the same
        # transaction, by a write that sends no signal: only the page's save
        # is observed.
        TextPlugin.objects.filter(pk=chairs.pk).update(page=news)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The plugin followed the page at its save, so its group is
    # replaced at the commit, as committed: on the page it moved to.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="News: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_deleting_a_topic_a_custom_extractor_depends_on_through_set_null_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, with a custom extractor reading its
    # topic's title when it has one: depends_on names the topic, its own
    # foreign key, SET_NULL on delete. Topic itself is not registered.
    @rag.register_extractor(Workshop, depends_on=["topic"])
    class WorkshopExtractor(BaseExtractor[Workshop]):
        def extract(self, instance: Workshop) -> NormalizedDocument:
            if instance.topic is None:
                return self.build_document(instance, text=instance.title)
            return self.build_document(
                instance, text=f"{instance.topic.title}: {instance.title}"
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)

    # The delete sets the workshop's foreign key to null before the topic's
    # row goes: by post_delete, the workshop no longer points to the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Workshop's group as committed: the topic's text is
    # gone, only the Workshop's own title is left.
    assert _received_groups(built_outputs) == {
        f"testapp.workshop:{pottery.pk}": [
            NormalizedDocument(
                text="Pottery",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=pottery.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_plugin_a_custom_extractor_depends_on_in_reverse_replaces_the_page(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Page is registered, with a custom extractor reading its text
    # plugins: depends_on names the reverse relation, whose saves change its
    # documents. TextPlugin itself is not registered.
    @rag.register_extractor(Page, depends_on=["text_plugins"])
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> NormalizedDocument:
            bodies = [plugin.body for plugin in instance.text_plugins.order_by("pk")]
            return self.build_document(
                instance, text="\n\n".join([instance.title, *bodies])
            )

    # Created outside the captured callbacks: the commit callback of the
    # Page's own save never runs, so only the plugin's save below is observed.
    page = Page.objects.create(title="About us", slug="about-us")

    with django_capture_on_commit_callbacks(execute=True):
        TextPlugin.objects.create(page=page, body="We build chairs by hand.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Page's group, with the plugin's text after its title.
    assert _received_groups(built_outputs) == {
        f"testapp.page:{page.pk}": [
            NormalizedDocument(
                text="About us\n\nWe build chairs by hand.",
                source_app_label="testapp",
                source_model="page",
                source_pk=page.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_remark_a_custom_extractor_depends_on_without_related_name_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Note is registered, with a custom extractor reading its
    # remarks: depends_on names the reverse relation by its default accessor,
    # "remark_set", while the relation's query name is "remark". Remark itself
    # is not registered.
    @rag.register_extractor(Note, depends_on=["remark_set"])
    class NoteExtractor(BaseExtractor[Note]):
        def extract(self, instance: Note) -> NormalizedDocument:
            bodies = [remark.body for remark in instance.remark_set.order_by("pk")]
            return self.build_document(
                instance, text="\n\n".join([instance.title, *bodies])
            )

    # Created outside the captured callbacks: the commit callback of the
    # Note's own save never runs, so only the remark's save below is observed.
    note = Note.objects.create(title="Workshop rules", body="Wear goggles.")

    with django_capture_on_commit_callbacks(execute=True):
        Remark.objects.create(note=note, body="Gloves too.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Note's group, with the remark's text after its title.
    assert _received_groups(built_outputs) == {
        f"testapp.note:{note.pk}": [
            NormalizedDocument(
                text="Workshop rules\n\nGloves too.",
                source_app_label="testapp",
                source_model="note",
                source_pk=note.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_profile_a_custom_extractor_depends_on_by_its_accessor_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Supplier is registered, with a custom extractor reading its
    # profile: depends_on names the reverse one-to-one by its accessor,
    # "profile", while the relation's query name is "supplier_profile".
    # SupplierProfile itself is not registered.
    @rag.register_extractor(Supplier, depends_on=["profile"])
    class SupplierExtractor(BaseExtractor[Supplier]):
        def extract(self, instance: Supplier) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.name}\n\n{instance.profile.body}"
            )

    # Created outside the captured callbacks: the commit callback of the
    # Supplier's own save never runs, so only the profile's save below is
    # observed.
    birch = Supplier.objects.create(name="Birch Mill")

    with django_capture_on_commit_callbacks(execute=True):
        SupplierProfile.objects.create(supplier=birch, body="Kiln-dried boards.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Supplier's group, with the profile's text after its
    # name.
    assert _received_groups(built_outputs) == {
        f"testapp.supplier:{birch.pk}": [
            NormalizedDocument(
                text="Birch Mill\n\nKiln-dried boards.",
                source_app_label="testapp",
                source_model="supplier",
                source_pk=birch.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_category_a_custom_extractor_depends_on_two_links_deep_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, with a custom extractor reading its
    # product's category's name: depends_on names the path through two foreign
    # keys, whose last link's saves change its documents. Neither Product nor
    # Category is registered.
    @rag.register_extractor(Offer, depends_on=["product__category"])
    class OfferExtractor(BaseExtractor[Offer]):
        def extract(self, instance: Offer) -> NormalizedDocument:
            return self.build_document(
                instance,
                text=f"{instance.title}: {instance.product.category.name}",
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_plain_desk_lamp(lighting)
    rake = Product.objects.create(
        name="Rake",
        description="Wooden.",
        price="14.50",
        category=garden,
    )
    spring = Offer.objects.create(title="Spring sale", product=lamp)
    # An offer of a product in another category: the save below does not
    # change its group.
    Offer.objects.create(title="Autumn sale", product=rake)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the offer of a product in the saved category,
    # with the category's new name.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{spring.pk}": [
            NormalizedDocument(
                text="Spring sale: Lamps",
                source_app_label="testapp",
                source_model="offer",
                source_pk=spring.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_category_a_custom_extractor_depends_on_via_parent_link_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the FeaturedProduct is registered, with a custom extractor reading
    # its category's name: depends_on names the path through the implicit
    # parent link, product_ptr, a forward one-to-one to Product, then
    # Product's foreign key to Category. Neither Product nor Category is
    # registered.
    @rag.register_extractor(FeaturedProduct, depends_on=["product_ptr__category"])
    class FeaturedProductExtractor(BaseExtractor[FeaturedProduct]):
        def extract(self, instance: FeaturedProduct) -> NormalizedDocument:
            return self.build_document(
                instance,
                text=f"{instance.tagline}: {instance.category.name}",
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_desk_lamp(lighting)
    # A featured product in another category: the save below does not change
    # its group.
    FeaturedProduct.objects.create(
        name="Rake",
        description="Wooden.",
        price="14.50",
        category=garden,
        tagline="Gather every leaf",
    )

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the featured product in the saved category,
    # with the category's new name.
    assert _received_groups(built_outputs) == {
        f"testapp.featuredproduct:{lamp.pk}": [
            NormalizedDocument(
                text="Light up your work: Lamps",
                source_app_label="testapp",
                source_model="featuredproduct",
                source_pk=lamp.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_category_a_custom_extractor_depends_on_then_parent_link_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Spotlight is registered, with a custom extractor reading its
    # featured product's category's name: depends_on names the path through
    # its foreign key to FeaturedProduct, then the implicit parent link,
    # product_ptr, a forward one-to-one to Product, then Product's foreign key
    # to Category. Neither FeaturedProduct, Product nor Category is registered.
    @rag.register_extractor(
        Spotlight, depends_on=["featured_product__product_ptr__category"]
    )
    class SpotlightExtractor(BaseExtractor[Spotlight]):
        def extract(self, instance: Spotlight) -> NormalizedDocument:
            return self.build_document(
                instance,
                text=f"{instance.title}: {instance.featured_product.category.name}",
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_desk_lamp(lighting)
    rake = FeaturedProduct.objects.create(
        name="Rake",
        description="Wooden.",
        price="14.50",
        category=garden,
        tagline="Gather every leaf",
    )
    window = Spotlight.objects.create(title="Shop window", featured_product=lamp)
    # A spotlight on a featured product in another category: the save below
    # does not change its group.
    Spotlight.objects.create(title="Front page", featured_product=rake)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the spotlight on a featured product in the
    # saved category, with the category's new name.
    assert _received_groups(built_outputs) == {
        f"testapp.spotlight:{window.pk}": [
            NormalizedDocument(
                text="Shop window: Lamps",
                source_app_label="testapp",
                source_model="spotlight",
                source_pk=window.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_topic_a_custom_extractor_depends_on_by_many_to_many_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, with a custom extractor reading the
    # titles of its topics: depends_on names its own many-to-many ``topics``,
    # whose saves change its documents. Topic itself is not registered.
    @rag.register_extractor(Course, depends_on=["topics"])
    class CourseExtractor(BaseExtractor[Course]):
        def extract(self, instance: Course) -> NormalizedDocument:
            titles = [topic.title for topic in instance.topics.order_by("pk")]
            return self.build_document(
                instance, text=f"{instance.title}: {', '.join(titles)}"
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's save below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    with django_capture_on_commit_callbacks(execute=True):
        woodworking.title = "Joinery"
        woodworking.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Course's group, with the topic's new title.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics: Joinery",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_course_a_custom_extractor_of_topic_depends_on_replaces_the_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, with a custom extractor reading the titles
    # of its courses: depends_on names its reverse many-to-many ``courses``,
    # whose links change its documents. Course itself is not registered.
    @rag.register_extractor(Topic, depends_on=["courses"])
    class TopicExtractor(BaseExtractor[Topic]):
        def extract(self, instance: Topic) -> NormalizedDocument:
            titles = [course.title for course in instance.courses.order_by("pk")]
            return self.build_document(
                instance, text=f"{instance.title}: {', '.join(titles)}"
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    # Added from the course's side: Django sends m2m_changed with the course as
    # its instance, and the topic among the primary keys it names.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Topic's group, with the added course's title.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking: Woodworking basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_course_a_custom_extractor_of_topic_depends_on_replaces_the_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, with a custom extractor reading the titles
    # of its courses: depends_on names its reverse many-to-many ``courses``,
    # whose saves change its documents. Course itself is not registered.
    @rag.register_extractor(Topic, depends_on=["courses"])
    class TopicExtractor(BaseExtractor[Topic]):
        def extract(self, instance: Topic) -> NormalizedDocument:
            titles = [course.title for course in instance.courses.order_by("pk")]
            return self.build_document(
                instance, text=f"{instance.title}: {', '.join(titles)}"
            )

    # Created and linked outside the captured callbacks: the commit callbacks
    # of these saves and of the add never run, so only the course's save below
    # is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    with django_capture_on_commit_callbacks(execute=True):
        basics.title = "Joinery basics"
        basics.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Topic's group, with the course's new title.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking: Joinery basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_deleting_a_course_a_custom_extractor_of_topic_depends_on_replaces_the_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, with a custom extractor reading the titles
    # of its courses: depends_on names its reverse many-to-many ``courses``,
    # whose deletes change its documents. Course itself is not registered.
    @rag.register_extractor(Topic, depends_on=["courses"])
    class TopicExtractor(BaseExtractor[Topic]):
        def extract(self, instance: Topic) -> NormalizedDocument:
            titles = [course.title for course in instance.courses.order_by("pk")]
            return self.build_document(
                instance, text=f"{instance.title}: {', '.join(titles)}"
            )

    # Created and linked outside the captured callbacks: the commit callbacks
    # of these saves and adds never run, so only the course's delete below is
    # observed. The topic is covered by two courses, so that its group keeps
    # the one left.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    joinery = Course.objects.create(title="Joinery")
    basics.topics.add(woodworking)
    joinery.topics.add(woodworking)

    # The delete removes the course's link to the topic before the course's
    # row goes: by post_delete, the topic no longer has the course.
    with django_capture_on_commit_callbacks(execute=True):
        basics.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Topic's group as committed: the deleted course's
    # title is gone, only the other course's title is left.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking: Joinery",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_page_empties_the_groups_of_depending_plugins_get_queryset_leaves_out(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the TextPlugin is registered, with a custom extractor reading its
    # page's title and depending on the page, whose get_queryset() leaves out
    # the plugins with an empty body. Page itself is not registered.
    @rag.register_extractor(TextPlugin, depends_on=["page"])
    class TextPluginExtractor(BaseExtractor[TextPlugin]):
        def get_queryset(self, queryset: QuerySet[TextPlugin]) -> QuerySet[TextPlugin]:
            return queryset.exclude(body="")

        def extract(self, instance: TextPlugin) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.page.title}: {instance.body}"
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the page's save below is observed.
    about = Page.objects.create(title="About us", slug="about-us")
    chairs = TextPlugin.objects.create(page=about, body="We build chairs by hand.")
    blank = TextPlugin.objects.create(page=about, body="")

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Our workshop"
        about.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The kept plugin's group, with the page's new title, and
    # the left-out plugin's group empty, so the output drops what it held.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="Our workshop: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
            ),
        ],
        f"testapp.textplugin:{blank.pk}": [],
    }


@pytest.mark.django_db(transaction=True)
def test_a_category_followed_by_foreign_key_saved_with_no_output_writes_no_row() -> (
    None
):
    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # No transaction around the save (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the save must fail before its
    # INSERT does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        Category.objects.create(name="Lighting")

    assert not Category.objects.exists()


def _save_with_signals_off(settings: Settings, category: Category) -> None:
    """Turn the signals off, then save the category the regular way."""
    settings.MODEL_RAG_SIGNALS = False
    category.save()


def _save_as_loaddata_does(settings: Settings, category: Category) -> None:
    """Save the category's row as loaddata saves a fixture: a raw save.

    The signals stay on: only the raw save can keep it from syncing.
    """
    fixture = [
        {"model": "testapp.category", "pk": category.pk, "fields": {"name": "Lamps"}}
    ]
    for deserialized in serializers.deserialize("python", fixture):
        deserialized.save()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "save",
    [
        pytest.param(_save_with_signals_off, id="signals_off"),
        pytest.param(_save_as_loaddata_does, id="raw_save"),
    ],
)
def test_a_category_followed_by_foreign_key_saved_unsynced_costs_nothing_more(
    save: Callable[[Settings, Category], None],
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # A working output, so that only the setting or the raw save can keep the
    # save from sending anything.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    lighting = Category.objects.create(name="Lighting")
    _create_a_plain_desk_lamp(lighting)

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits. With signals on and a regular save, the save of
    # an existing row would look its followers up: one query more.
    with (
        django_assert_num_queries(1) as queries,
        django_capture_on_commit_callbacks(execute=True) as callbacks,
    ):
        lighting.name = "Lamps"
        save(settings, lighting)

    # The row is written, yet the only query is the save's UPDATE: no lookup of
    # followers. Nothing is deferred to the commit, and no output is built.
    assert Category.objects.get(pk=lighting.pk).name == "Lamps"
    statements = _statements(queries)
    assert statements == ["UPDATE"]
    assert callbacks == []
    assert built_outputs == []


@pytest.mark.django_db
def test_an_output_failing_on_the_followers_of_a_category_logs_the_category_saved(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the categories' saves below are observed.
    lighting = Category.objects.create(name="Lighting")
    # Two followers, so that the message cannot name the one follower there is.
    lamp = _create_a_plain_desk_lamp(lighting)
    _create_a_bulb(lighting)
    tools = Category.objects.create(name="Tools")
    hammer = _create_a_hammer(tools)

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.product:{lamp.pk}"},
    }

    # The failing category is saved first, so that its failure comes before the
    # other category's followers are sent. An error escaping the commit
    # callbacks would fail the test: the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting.name = "Lamps"
        lighting.save()
        tools.name = "Hardware"
        tools.save()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the followers' model and the category saved, however
    # many followers it has, and carries the error itself.
    assert record.getMessage() == (
        f"Syncing testapp.product instances that follow "
        f"testapp.category:{lighting.pk} failed"
    )
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], FailingReplaceError)
    # The other category's follower still reaches an output.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{hammer.pk}": [
                NormalizedDocument(
                    text="Hammer\n\nHardware",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=hammer.pk,
                    title="Hammer",
                    url=f"/products/{hammer.pk}/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_an_output_failing_on_the_followers_of_a_category_proxy_logs_the_category(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: neither Category nor its proxy is.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's save below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp = _create_a_plain_desk_lamp(lighting)

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.product:{lamp.pk}"},
    }

    # Django sends post_save with the proxy as its sender, not Category. An
    # error escaping the commit callbacks would fail the test: the commit
    # itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lamps = CategoryProxy.objects.get(pk=lighting.pk)
        lamps.name = "Lamps"
        lamps.save()

    [record] = _package_log_records(caplog)
    # The record names the row saved by its concrete model's key, the one its
    # source keys use, not by the proxy's.
    assert record.getMessage() == (
        f"Syncing testapp.product instances that follow "
        f"testapp.category:{lighting.pk} failed"
    )


@pytest.mark.django_db
def test_an_output_failing_on_the_followers_of_a_deleted_topic_proxy_logs_the_topic(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: neither Topic nor its proxy is. The
    # Workshop outlives its topic, so its group is replaced at the commit.
    rag.register(Workshop, follow=["topic"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)
    woodworking_pk = woodworking.pk

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.workshop:{pottery.pk}"},
    }

    # Django sends pre_delete and post_delete with the proxy as their sender,
    # not Topic. An error escaping the commit callbacks would fail the test:
    # the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        TopicProxy.objects.get(pk=woodworking_pk).delete()

    [record] = _package_log_records(caplog)
    # The record names the row deleted by its concrete model's key, the one its
    # source keys use, not by the proxy's.
    assert record.getMessage() == (
        f"Syncing testapp.workshop instances that follow "
        f"testapp.topic:{woodworking_pk} failed"
    )


@pytest.mark.django_db
def test_deleting_a_topic_followed_through_set_null_replaces_the_workshops_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    rag.register(Workshop, follow=["topic"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)

    # The delete sets the workshop's foreign key to null before the topic's
    # row goes: by post_delete, the workshop no longer points to the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Workshop's group as committed: the topic's text is gone, only the
    # Workshop's own title is left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.workshop:{pottery.pk}": [
                NormalizedDocument(
                    text="Pottery",
                    source_app_label="testapp",
                    source_model="workshop",
                    source_pk=pottery.pk,
                    title="Pottery",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_a_topic_saved_then_deleted_replaces_no_group_of_the_workshops_with_no_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    rag.register(Workshop, follow=["topic"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's save and delete below are observed.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)
    # Its nullable foreign key is null: it never followed the topic.
    Workshop.objects.create(title="Ceramics", topic=None)

    # The delete clears the topic's primary key before the commit, where the
    # save looks its followers up.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.title = "Joinery"
        woodworking.save()
        woodworking.delete()

    # Only the group of the workshop that pointed to the topic, as committed,
    # merged across replace calls: whether the save and the delete send it once
    # or twice is not what this test is about. The workshop with no topic is
    # not sent.
    assert _received_groups(built_outputs) == {
        f"testapp.workshop:{pottery.pk}": [
            NormalizedDocument(
                text="Pottery",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=pottery.pk,
                title="Pottery",
            ),
        ],
    }


@pytest.mark.django_db
def test_deleting_a_topic_read_two_links_deep_through_set_null_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Session is registered, reading its workshop's topic's title
    # through a lookup path two links deep, the last one SET_NULL on delete,
    # with no follow: neither Workshop nor Topic is registered, so the topic
    # reaches the session only through the path.
    rag.register(Session, fields=["title", "workshop__topic__title"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    glazing = Topic.objects.create(
        summary="Colours and kilns.", title="Glazing", slug="glazing"
    )
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)
    ceramics = Workshop.objects.create(title="Ceramics", topic=glazing)
    morning = Session.objects.create(title="Morning", workshop=pottery)
    Session.objects.create(title="Evening", workshop=ceramics)

    # The delete sets the workshop's foreign key to null before the topic's
    # row goes: by post_delete, the path from the session no longer reaches
    # the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The session's group as committed: the topic's title is gone, only the
    # session's own title is left. The session of a workshop on another topic
    # is not sent.
    assert _replaced(built_outputs) == [
        {
            f"testapp.session:{morning.pk}": [
                NormalizedDocument(
                    text="Morning",
                    source_app_label="testapp",
                    source_model="session",
                    source_pk=morning.pk,
                    title="Morning",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_deleting_a_category_whose_following_products_cascade_sends_only_their_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key, CASCADE on delete: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's delete below is observed.
    lighting = Category.objects.create(name="Lighting")
    lamp_pk = _create_a_plain_desk_lamp(lighting).pk
    bulb_pk = _create_a_bulb(lighting).pk

    # The products are deleted with the category by cascade: they are found as
    # its followers before the delete, yet no longer exist at the commit.
    with django_capture_on_commit_callbacks(execute=True):
        lighting.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The products' empty groups, merged across replace calls: whether they
    # come in one call or one per product is not what this test is about.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{lamp_pk}": [],
        f"testapp.product:{bulb_pk}": [],
    }
    # Each group sent once: no replacement of a product's group follows its
    # empty one.
    sent_source_keys = [
        source_key for groups in _replaced(built_outputs) for source_key in groups
    ]
    assert sorted(sent_source_keys) == sorted(
        [f"testapp.product:{lamp_pk}", f"testapp.product:{bulb_pk}"]
    )


@pytest.mark.django_db
def test_deleting_through_a_proxy_of_a_topic_followed_through_set_null_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: neither Topic nor its proxy is.
    rag.register(Workshop, follow=["topic"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)

    # Django sends pre_delete and post_delete with the proxy as their sender,
    # not Topic.
    with django_capture_on_commit_callbacks(execute=True):
        TopicProxy.objects.get(pk=woodworking.pk).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Workshop's group as committed: the topic's text is gone, only the
    # Workshop's own title is left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.workshop:{pottery.pk}": [
                NormalizedDocument(
                    text="Pottery",
                    source_app_label="testapp",
                    source_model="workshop",
                    source_pk=pottery.pk,
                    title="Pottery",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_deleting_a_theme_followed_through_a_set_null_key_to_its_proxy_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Meetup is registered, following its theme through its own
    # foreign key, SET_NULL on delete, whose model is the proxy ThemeProxy:
    # neither Theme nor its proxy is.
    rag.register(Meetup, follow=["theme"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the theme's delete below is observed.
    joinery = Theme.objects.create(name="Joinery")
    evening = Meetup.objects.create(title="Evening meetup", theme_id=joinery.pk)

    # A plain Theme, not its proxy: Django sends pre_delete and post_delete
    # with Theme as their sender, while the Meetup's foreign key names the
    # proxy.
    with django_capture_on_commit_callbacks(execute=True):
        joinery.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Meetup's group as committed: the theme's text is gone, only the
    # Meetup's own title is left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.meetup:{evening.pk}": [
                NormalizedDocument(
                    text="Evening meetup",
                    source_app_label="testapp",
                    source_model="meetup",
                    source_pk=evening.pk,
                    title="Evening meetup",
                ),
            ],
        }
    ]


@pytest.mark.django_db(transaction=True)
def test_a_followed_topic_deleted_with_no_output_fails_and_signals_off_costs_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Created before Workshop is registered: with no MODEL_RAG_OUTPUT, their
    # own saves would fail otherwise.
    woodworking = _create_the_woodworking_topic()
    pottery = Workshop.objects.create(title="Pottery", topic=woodworking)

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    rag.register(Workshop, follow=["topic"])

    # No transaction around the delete (transaction=True): the delete runs in
    # its own, which commits as it ends, so it must fail before then.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        woodworking.delete()

    # The delete is rolled back: the topic's row is there, and the workshop
    # still points to it.
    assert Topic.objects.filter(pk=woodworking.pk).exists()
    assert Workshop.objects.get(pk=pottery.pk).topic_id == woodworking.pk

    settings.MODEL_RAG_SIGNALS = False
    # A working output, so that only the setting can keep the same delete from
    # sending to it.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The delete commits as it ends, running any commit callback it deferred.
    # With signals on, the delete would look its followers up first: one query
    # more than the delete's own.
    with django_assert_num_queries(6) as queries:
        woodworking.delete()

    # The rows are deleted and nulled, yet the only queries are the delete's
    # own, in its transaction: the topic's lessons and course links, the
    # workshop's foreign key, the topic itself. No lookup of followers, and no
    # output is built, even after the commit.
    assert not Topic.objects.exists()
    assert Workshop.objects.get(pk=pottery.pk).topic_id is None
    statements = _statements(queries)
    assert statements == ["BEGIN", "DELETE", "DELETE", "UPDATE", "DELETE", "COMMIT"]
    assert built_outputs == []


@pytest.mark.django_db
def test_saving_a_registered_instance_also_followed_replaces_its_group_and_the_other(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # TextPlugin is registered itself, and also followed by its Page.
    _register_pages_following_their_plugins()
    rag.register(TextPlugin, fields=["body"])

    # Created outside the captured callbacks: the commit callback of the
    # Page's own save never runs, so only the plugin's save below is observed.
    page = Page.objects.create(title="About us", slug="about-us")

    with django_capture_on_commit_callbacks(execute=True):
        plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per group is not what this test is about.
    received = _received_groups(built_outputs)
    # Both groups as committed: the plugin's own, with its body, and the
    # Page's, with the plugin's text after the Page's own title.
    assert received == {
        f"testapp.textplugin:{plugin.pk}": [
            NormalizedDocument(
                text="We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=plugin.pk,
                # Without a title field, the first declared field's text.
                title="We build chairs by hand.",
            ),
        ],
        f"testapp.page:{page.pk}": [
            NormalizedDocument(
                text="About us\n\nWe build chairs by hand.",
                source_app_label="testapp",
                source_model="page",
                source_pk=page.pk,
                title="About us",
                url="/pages/about-us/",
            ),
        ],
    }


@pytest.mark.django_db
def test_moving_a_followed_instance_to_another_follower_replaces_the_groups_of_both(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the plugin's move below is observed.
    about = Page.objects.create(title="About us", slug="about-us")
    workshop = Page.objects.create(title="Our workshop", slug="our-workshop")
    moved_plugin = TextPlugin.objects.create(
        page=about, body="We build chairs by hand."
    )
    TextPlugin.objects.create(page=about, body="We ship worldwide.")

    with django_capture_on_commit_callbacks(execute=True):
        moved_plugin.page = workshop
        moved_plugin.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per Page is not what this test is about.
    received = _received_groups(built_outputs)
    # Both Pages' groups as committed: the old Page keeps only its remaining
    # plugin's text, the new Page gains the moved plugin's text.
    assert received == {
        f"testapp.page:{about.pk}": [
            NormalizedDocument(
                text="About us\n\nWe ship worldwide.",
                source_app_label="testapp",
                source_model="page",
                source_pk=about.pk,
                title="About us",
                url="/pages/about-us/",
            ),
        ],
        f"testapp.page:{workshop.pk}": [
            NormalizedDocument(
                text="Our workshop\n\nWe build chairs by hand.",
                source_app_label="testapp",
                source_model="page",
                source_pk=workshop.pk,
                title="Our workshop",
                url="/pages/our-workshop/",
            ),
        ],
    }


@pytest.mark.django_db
def test_moving_a_followed_instance_built_with_an_existing_pk_replaces_both_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the plugin's move below is observed.
    about = Page.objects.create(title="About us", slug="about-us")
    workshop = Page.objects.create(title="Our workshop", slug="our-workshop")
    existing_plugin = TextPlugin.objects.create(
        page=about, body="We build chairs by hand."
    )
    TextPlugin.objects.create(page=about, body="We ship worldwide.")

    # Not loaded: built anew with the existing row's primary key, so Django
    # marks it as being added, yet saves it as an UPDATE of that row.
    moved_plugin = TextPlugin(
        pk=existing_plugin.pk, page=workshop, body="We build chairs by hand."
    )
    with django_capture_on_commit_callbacks(execute=True):
        moved_plugin.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The row was updated, not duplicated: the move is the one being specified.
    assert TextPlugin.objects.filter(pk=existing_plugin.pk, page=workshop).exists()
    # Merged across replace calls: whether the groups come in one call or one
    # per Page is not what this test is about.
    received = _received_groups(built_outputs)
    # Both Pages' groups as committed: the old Page keeps only its remaining
    # plugin's text, the new Page gains the moved plugin's text.
    assert received == {
        f"testapp.page:{about.pk}": [
            NormalizedDocument(
                text="About us\n\nWe ship worldwide.",
                source_app_label="testapp",
                source_model="page",
                source_pk=about.pk,
                title="About us",
                url="/pages/about-us/",
            ),
        ],
        f"testapp.page:{workshop.pk}": [
            NormalizedDocument(
                text="Our workshop\n\nWe build chairs by hand.",
                source_app_label="testapp",
                source_model="page",
                source_pk=workshop.pk,
                title="Our workshop",
                url="/pages/our-workshop/",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_change_left_unsaved_after_a_move_does_not_change_the_groups_replaced(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # A Product is followed by its Category, through the reverse relation
    # ``products``, and by its reviews, through their own foreign key: the
    # latter makes its followers be looked up at the commit. Product itself is
    # not registered.
    rag.register(Category, follow=["products"])
    rag.register(Review, follow=["product"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the product's move below is observed.
    lighting = Category.objects.create(name="Lighting")
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    lamp = _create_a_plain_desk_lamp(lighting)
    review = Review.objects.create(title="Sturdy", product=lamp)

    with django_capture_on_commit_callbacks(execute=True):
        lamp.category = tools
        lamp.save()
        # Changed in memory after the save, and never saved: the row the
        # commit sees still belongs to the Tools category.
        lamp.category = garden
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The groups of the categories the product was saved out of
    # and into, and of its review, all as committed; no group of the category
    # it was only given in memory.
    assert _received_groups(built_outputs) == {
        f"testapp.category:{lighting.pk}": [
            NormalizedDocument(
                text="Lighting",
                source_app_label="testapp",
                source_model="category",
                source_pk=lighting.pk,
                title="Lighting",
            ),
        ],
        f"testapp.category:{tools.pk}": [
            NormalizedDocument(
                text="Tools\n\nDesk lamp\n\nA lamp for the desk.\n\nNew",
                source_app_label="testapp",
                source_model="category",
                source_pk=tools.pk,
                title="Tools",
            ),
        ],
        f"testapp.review:{review.pk}": [
            NormalizedDocument(
                text="Sturdy\n\nDesk lamp\n\nA lamp for the desk.\n\nNew",
                source_app_label="testapp",
                source_model="review",
                source_pk=review.pk,
                title="Sturdy",
            ),
        ],
    }


@pytest.mark.django_db
def test_updating_a_followed_instance_in_place_replaces_the_group_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the plugin's update below is observed.
    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    # The plugin stays on its Page: the Page it had before the save is the
    # Page it has after.
    with django_capture_on_commit_callbacks(execute=True):
        plugin.body = "We ship worldwide."
        plugin.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # One replace call, not one for the Page before the save and another for
    # the same Page after it.
    assert _replaced(built_outputs) == [
        {
            f"testapp.page:{page.pk}": [
                NormalizedDocument(
                    text="About us\n\nWe ship worldwide.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=page.pk,
                    title="About us",
                    url="/pages/about-us/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_or_deleting_through_a_proxy_of_a_followed_model_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Neither TextPlugin nor its proxy is registered.
    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the saves and deletes below are observed.
    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    # Django sends post_save with the proxy as its sender, not TextPlugin.
    with django_capture_on_commit_callbacks(execute=True):
        TextPluginProxy.objects.create(page=page, body="We ship worldwide.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Page's group as committed, with the text of both plugins.
    assert _replaced(built_outputs) == [
        {
            f"testapp.page:{page.pk}": [
                NormalizedDocument(
                    text="About us\n\nWe build chairs by hand.\n\nWe ship worldwide.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=page.pk,
                    title="About us",
                    url="/pages/about-us/",
                ),
            ],
        }
    ]
    built_outputs.clear()

    # Django sends post_delete with the proxy as its sender, not TextPlugin.
    with django_capture_on_commit_callbacks(execute=True):
        TextPluginProxy.objects.get(pk=plugin.pk).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Page's group as committed: the deleted plugin's text is gone.
    assert _replaced(built_outputs) == [
        {
            f"testapp.page:{page.pk}": [
                NormalizedDocument(
                    text="About us\n\nWe ship worldwide.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=page.pk,
                    title="About us",
                    url="/pages/about-us/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_multi_table_child_of_a_followed_model_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Neither Track nor BonusTrack is registered.
    rag.register(Album, follow=["tracks"])

    # Created outside the captured callbacks: the commit callback of the
    # Album's own save never runs, so only the bonus track's save is observed.
    album = Album.objects.create(title="Abbey Road")

    # Django sends post_save with BonusTrack as its sender, not Track, though
    # the save writes a Track row the Album follows.
    with django_capture_on_commit_callbacks(execute=True):
        BonusTrack.objects.create(
            album=album, title="Her Majesty", note="Hidden after the last track"
        )
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Album's group as committed, with the track's title after its own.
    assert _replaced(built_outputs) == [
        {
            f"testapp.album:{album.pk}": [
                NormalizedDocument(
                    text="Abbey Road\n\nHer Majesty",
                    source_app_label="testapp",
                    source_model="album",
                    source_pk=album.pk,
                    title="Abbey Road",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_an_instance_followed_by_two_models_replaces_the_group_of_each(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # An Exhibit is followed twice, through each of its foreign keys: by its
    # Showroom and by its Category. Exhibit itself is not registered.
    rag.register(Showroom, follow=["exhibits"])
    rag.register(Category, follow=["exhibits"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the exhibit's save below is observed.
    north = Showroom.objects.create(name="North hall")
    tools = Category.objects.create(name="Tools")

    with django_capture_on_commit_callbacks(execute=True):
        Exhibit.objects.create(showroom=north, label="Hammer", category=tools)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per follower is not what this test is about.
    received = _received_groups(built_outputs)
    # Both followers' groups as committed, each with the exhibit's label after
    # its own name.
    assert received == {
        f"testapp.showroom:{north.pk}": [
            NormalizedDocument(
                text="North hall\n\nHammer",
                source_app_label="testapp",
                source_model="showroom",
                source_pk=north.pk,
                title="North hall",
            ),
        ],
        f"testapp.category:{tools.pk}": [
            NormalizedDocument(
                text="Tools\n\nHammer",
                source_app_label="testapp",
                source_model="category",
                source_pk=tools.pk,
                title="Tools",
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_followed_reverse_one_to_one_replaces_the_group_that_follows_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Supplier is registered, following its profile by the reverse
    # one-to-one accessor ``profile``: SupplierProfile itself is not.
    rag.register(Supplier, follow=["profile"])

    # Created outside the captured callbacks: the commit callback of the
    # Supplier's own save never runs, so only the profile's save below is
    # observed.
    birch = Supplier.objects.create(name="Birch Mill")

    with django_capture_on_commit_callbacks(execute=True):
        SupplierProfile.objects.create(supplier=birch, body="Kiln-dried boards.")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Supplier's group, with the profile's text after the Supplier's name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.supplier:{birch.pk}": [
                NormalizedDocument(
                    text="Birch Mill\n\nKiln-dried boards.",
                    source_app_label="testapp",
                    source_model="supplier",
                    source_pk=birch.pk,
                    title="Birch Mill",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_followed_instance_linked_by_a_unique_column_replaces_the_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Warehouse is registered, following its shelves: Shelf itself is
    # not.
    rag.register(Warehouse, follow=["shelves"])

    # Created outside the captured callbacks: the commit callback of the
    # Warehouse's own save never runs, so only the shelf's save below is
    # observed.
    north = Warehouse.objects.create(name="North depot", code="north")

    with django_capture_on_commit_callbacks(execute=True):
        # The shelf's foreign key holds the Warehouse's code, "north", not its
        # primary key.
        Shelf.objects.create(warehouse=north, label="Timber")
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Warehouse's group, under its primary key, not its code, with the
    # shelf's label after the Warehouse's name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.warehouse:{north.pk}": [
                NormalizedDocument(
                    text="North depot\n\nTimber",
                    source_app_label="testapp",
                    source_model="warehouse",
                    source_pk=north.pk,
                    title="North depot",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_a_null_foreign_key_to_a_nullable_unique_column_sends_nothing_for_a_null_row(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Depot is registered, following its bins: Bin itself is not.
    rag.register(Depot, follow=["bins"])

    # Created outside the captured callbacks: the commit callback of the
    # Depot's own save never runs, so only the bin's save below is observed.
    # Its code is null, as the bin's foreign key will be.
    Depot.objects.create(name="Unnamed depot", code=None)

    # The commit callbacks run, and an error escaping them would fail the test.
    with django_capture_on_commit_callbacks(execute=True):
        # Its nullable foreign key is null: it points to no Depot, not even to
        # the one whose code is null, as the database never joins NULL to NULL.
        Bin.objects.create(depot=None, label="Spare parts")

    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_saving_a_followed_instance_with_no_follower_sends_nothing_and_logs_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its workshops: Workshop itself
    # is not.
    rag.register(Topic, follow=["workshops"])

    # The commit callbacks run, and an error escaping them would fail the test.
    with (
        caplog.at_level(logging.DEBUG, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Its nullable foreign key is null: no Topic follows this workshop.
        Workshop.objects.create(title="Open bench", topic=None)

    # Not even an output built, and nothing logged: there is no group to
    # replace, not a failure to report.
    assert built_outputs == []
    assert _package_log_records(caplog) == []


@pytest.mark.django_db
def test_saving_a_followed_instance_with_no_follower_defers_nothing_to_the_commit(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its workshops: Workshop itself
    # is not.
    rag.register(Topic, follow=["workshops"])

    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        # Its nullable foreign key is null: no Topic follows this workshop.
        Workshop.objects.create(title="Open bench", topic=None)

    # No follower, so no group to replace: nothing is even deferred to the
    # commit.
    assert callbacks == []


@pytest.mark.django_db
def test_saving_a_course_followed_by_reverse_many_to_many_replaces_the_topics_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the course's save below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    with django_capture_on_commit_callbacks(execute=True):
        basics.title = "Furniture making"
        basics.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Topic's group, with the course's new title after the Topic's own
    # title and summary.
    assert _replaced(built_outputs) == [
        {
            f"testapp.topic:{woodworking.pk}": [
                NormalizedDocument(
                    text="Woodworking\n\nJoints and finishes.\n\nFurniture making",
                    source_app_label="testapp",
                    source_model="topic",
                    source_pk=woodworking.pk,
                    title="Woodworking",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_topic_followed_by_forward_many_to_many_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's save below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    with django_capture_on_commit_callbacks(execute=True):
        woodworking.title = "Joinery"
        woodworking.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Course's group, with the topic's new title and its summary after the
    # Course's own title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nJoinery\n\nJoints and finishes.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_topic_to_a_course_following_its_topics_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Neither row is
    # saved again: the add writes only the link between them.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Course's group as committed: its own title, then the added topic's
    # title and summary.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nWoodworking\n\nJoints and finishes.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_course_to_a_topic_replaces_the_group_of_the_course_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Neither row is
    # saved again: the add writes only the link between them.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # A course already covering the topic: the add below does not change its
    # group.
    joinery = Course.objects.create(title="Joinery")
    joinery.topics.add(woodworking)

    # Added from the reverse side: Django sends m2m_changed with the topic as
    # its instance, and the course among the primary keys it names.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The added Course's group as committed: its own title, then the topic's
    # title and summary. No group of the course already covering the topic.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nWoodworking\n\nJoints and finishes.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_removing_a_topic_from_a_course_following_its_topics_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the remove below is observed. The
    # course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)

    # Neither row is saved again: the remove deletes only the link between
    # them.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.remove(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Course's group as committed: the removed topic's text is gone, only
    # the Course's own title and the other topic's text are left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nCarving\n\nKnives and gouges.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_removing_a_course_from_a_topic_replaces_the_group_of_the_course_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed.
    # The course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)
    # A course still covering the topic after the remove below: its group does
    # not change.
    joinery = Course.objects.create(title="Joinery")
    joinery.topics.add(woodworking)

    # Removed from the reverse side: Django sends m2m_changed with the topic as
    # its instance, and the course among the primary keys it names. Neither row
    # is saved again: the remove deletes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.remove(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The removed Course's group as committed: the topic's text is gone, only
    # the Course's own title and the other topic's text are left. No group of
    # the course still covering the topic.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nCarving\n\nKnives and gouges.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_clearing_the_topics_of_a_course_following_them_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the clear below is observed. The
    # course covers two topics, so that the clear removes more than one link.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)

    # Neither row is saved again: the clear deletes only the links of the
    # course.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Course's group as committed: no topic's text is left, only the
    # Course's own title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_clearing_the_courses_of_a_topic_replaces_the_group_of_each_course_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Two courses cover the topic: one covers another topic too, so that its
    # group keeps the one left, the other covers it alone.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)
    joinery = Course.objects.create(title="Joinery")
    joinery.topics.add(woodworking)
    # A course not covering the topic: the clear below does not change its
    # group.
    whittling = Course.objects.create(title="Whittling")
    whittling.topics.add(carving)

    # Cleared from the reverse side: Django sends m2m_changed with the topic as
    # its instance and no primary keys at all, so the courses covering it can
    # only be found before the clear. No row is saved again: the clear deletes
    # only the links of the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of both courses as committed, each without the topic's text,
    # and no group of the course that never covered it.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
        f"testapp.course:{joinery.pk}": [
            NormalizedDocument(
                text="Joinery",
                source_app_label="testapp",
                source_model="course",
                source_pk=joinery.pk,
                title="Joinery",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_topic_to_a_course_replaces_the_group_of_the_topic_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # A topic the course already covers: the add below does not change its
    # group.
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics.topics.add(carving)

    # Added from the course's side: Django sends m2m_changed with the course as
    # its instance, and the topic among the primary keys it names. Neither row
    # is saved again: the add writes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The added Topic's group as committed: its own title and summary, then the
    # course's title. No group of the topic the course already covered.
    assert _replaced(built_outputs) == [
        {
            f"testapp.topic:{woodworking.pk}": [
                NormalizedDocument(
                    text="Woodworking\n\nJoints and finishes.\n\nWoodworking basics",
                    source_app_label="testapp",
                    source_model="topic",
                    source_pk=woodworking.pk,
                    title="Woodworking",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_course_to_a_topic_following_its_courses_replaces_the_topics_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # Another topic the course already covers: the add below does not change
    # its group.
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics.topics.add(carving)

    # Added from the topic's side: Django sends m2m_changed with the topic as
    # its instance, and the course among the primary keys it names. Neither row
    # is saved again: the add writes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Topic's group as committed: its own title and summary, then the added
    # course's title. No group of the other topic the course covers.
    assert _replaced(built_outputs) == [
        {
            f"testapp.topic:{woodworking.pk}": [
                NormalizedDocument(
                    text="Woodworking\n\nJoints and finishes.\n\nWoodworking basics",
                    source_app_label="testapp",
                    source_model="topic",
                    source_pk=woodworking.pk,
                    title="Woodworking",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_course_to_a_topic_through_a_proxy_replaces_the_topics_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: neither Course nor TopicProxy is.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # The same Topic's row, read under the proxy's class.
    proxy_woodworking = TopicProxy.objects.get(pk=woodworking.pk)

    # Added from the proxy's side: Django sends m2m_changed with the proxy
    # instance as its instance, not a Topic.
    with django_capture_on_commit_callbacks(execute=True):
        proxy_woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Topic's group as committed: its own title and summary, then the added
    # course's title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.topic:{woodworking.pk}": [
                NormalizedDocument(
                    text="Woodworking\n\nJoints and finishes.\n\nWoodworking basics",
                    source_app_label="testapp",
                    source_model="topic",
                    source_pk=woodworking.pk,
                    title="Woodworking",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_clearing_the_courses_of_a_topic_through_a_proxy_replaces_the_courses_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: neither Topic nor TopicProxy is.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Two courses cover the topic: one covers another topic too, so that its
    # group keeps the one left, the other covers it alone.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)
    joinery = Course.objects.create(title="Joinery")
    joinery.topics.add(woodworking)
    # A course not covering the topic: the clear below does not change its
    # group.
    whittling = Course.objects.create(title="Whittling")
    whittling.topics.add(carving)
    # The same Topic's row, read under the proxy's class.
    proxy_woodworking = TopicProxy.objects.get(pk=woodworking.pk)

    # Cleared from the proxy's side: Django sends m2m_changed with the proxy
    # instance as its instance, not a Topic, and no primary keys at all.
    with django_capture_on_commit_callbacks(execute=True):
        proxy_woodworking.courses.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of both courses as committed, each without the topic's text,
    # and no group of the course that never covered it.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
        f"testapp.course:{joinery.pk}": [
            NormalizedDocument(
                text="Joinery",
                source_app_label="testapp",
                source_model="course",
                source_pk=joinery.pk,
                title="Joinery",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_two_topics_to_a_course_replaces_the_group_of_each_topic_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    # A topic the course already covers: the add below does not change its
    # group.
    finishing = Topic.objects.create(
        summary="Oils and waxes.", title="Finishing", slug="finishing"
    )
    basics.topics.add(finishing)

    # Two topics added at once from the course's side: Django sends one
    # m2m_changed with the course as its instance, and both topics among the
    # primary keys it names.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking, carving)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of each added Topic as committed, each ending with the course's
    # title. No group of the topic the course already covered.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking\n\nJoints and finishes.\n\nWoodworking basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
                title="Woodworking",
            ),
        ],
        f"testapp.topic:{carving.pk}": [
            NormalizedDocument(
                text="Carving\n\nKnives and gouges.\n\nWoodworking basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=carving.pk,
                title="Carving",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_guild_to_a_craftsman_replaces_the_guilds_group_named_by_its_code(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Guild is registered, following its members through its own
    # many-to-many ``members``: Craftsman itself is not.
    rag.register(Guild, fields=["name"], follow=["members"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed. A
    # guild's code, a slug, can never equal its integer primary key.
    carpenter = Craftsman.objects.create(name="Carpenter")
    north = Guild.objects.create(name="North guild", code="north")
    # A guild the craftsman already belongs to: the add below does not change
    # its group.
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(carpenter)

    # Added from the craftsman's side: Django sends m2m_changed with the
    # craftsman as its instance, and names the guild by the column
    # Membership.guild points to, its code, not by its primary key. Neither row
    # is saved again: the add writes only the membership between them.
    with django_capture_on_commit_callbacks(execute=True):
        carpenter.guilds.add(north)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The added Guild's group as committed: its name, then its new member's
    # name. No group of the guild the craftsman already belonged to.
    assert _replaced(built_outputs) == [
        {
            f"testapp.guild:{north.pk}": [
                NormalizedDocument(
                    text="North guild\n\nCarpenter",
                    source_app_label="testapp",
                    source_model="guild",
                    source_pk=north.pk,
                    title="North guild",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_topic_to_a_course_neither_side_following_the_link_defers_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only what each side follows can keep the add
    # from deferring anything to the commit.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Both sides of the link are registered, neither following it: the Course
    # does not follow its topics, nor the Topic its courses.
    rag.register(Course)
    rag.register(Topic)

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Neither row is
    # saved again: the add writes only the link between them.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    # The commit callbacks run: one sending anything would reach the output.
    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        basics.topics.add(woodworking)

    # The link is part of neither group: nothing is deferred to the commit,
    # and nothing reaches the output.
    assert callbacks == []
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_adding_a_topic_to_a_course_following_its_topics_with_signals_off_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the setting can keep the add from sending
    # to it.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Neither row is
    # saved again: the add writes only the link between them.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    settings.MODEL_RAG_SIGNALS = False

    # The commit callbacks run: one sending anything would reach the output.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)

    # The link is written, yet nothing reaches the output.
    assert list(basics.topics.all()) == [woodworking]
    assert _replaced(built_outputs) == []


@pytest.mark.django_db(transaction=True)
def test_a_topic_added_to_a_course_following_it_with_no_output_writes_no_link() -> None:
    # Created before Course is registered: with no MODEL_RAG_OUTPUT, their own
    # saves would fail otherwise.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # No transaction around the add (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the add must fail before its
    # INSERT of the join row does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        basics.topics.add(woodworking)

    assert not basics.topics.exists()


@pytest.mark.django_db(transaction=True)
def test_a_following_course_added_to_a_topic_with_no_output_writes_no_link() -> None:
    # Created before Course is registered: with no MODEL_RAG_OUTPUT, their own
    # saves would fail otherwise.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Added from the topic's side: the course following the link is among the
    # rows the add names, not its instance. No transaction around the add
    # (transaction=True): in autocommit, each query commits as soon as it
    # runs, so the add must fail before its INSERT of the join row does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        woodworking.courses.add(basics)

    assert not basics.topics.exists()


@pytest.mark.django_db(transaction=True)
def test_a_topic_removed_from_a_course_following_it_with_no_output_keeps_the_link() -> (
    None
):
    # Created and linked before Course is registered: with no MODEL_RAG_OUTPUT,
    # their own saves and the add would fail otherwise.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # No transaction around the remove (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the remove must fail before its
    # DELETE of the join row does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        basics.topics.remove(woodworking)

    assert list(basics.topics.all()) == [woodworking]


@pytest.mark.django_db(transaction=True)
def test_the_topics_of_a_course_following_them_cleared_with_no_output_stay_linked() -> (
    None
):
    # Created and linked before Course is registered: with no MODEL_RAG_OUTPUT,
    # their own saves and the add would fail otherwise. The course covers two
    # topics, so that the clear would delete more than one link.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # No transaction around the clear (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the clear must fail before its
    # DELETE of the join rows does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        basics.topics.clear()

    assert set(basics.topics.all()) == {woodworking, carving}


@pytest.mark.django_db
def test_an_output_failing_on_a_course_added_to_a_topic_logs_the_course_and_the_topic(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Neither row is
    # saved again: the add writes only the link between them.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.course:{basics.pk}"},
    }

    # Added from the reverse side: Django sends m2m_changed with the topic as
    # its instance. An error escaping the commit callbacks would fail the test:
    # the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        woodworking.courses.add(basics)

    # The link is committed all the same.
    assert list(basics.topics.all()) == [woodworking]
    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the followers' model and the topic the add came from,
    # and carries the error.
    assert record.getMessage() == (
        f"Syncing testapp.course instances that follow "
        f"testapp.topic:{woodworking.pk} failed"
    )
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], FailingReplaceError)


@pytest.mark.django_db
def test_deleting_a_topic_followed_by_forward_many_to_many_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many ``topics``: Topic itself is not.
    rag.register(Course, follow=["topics"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's delete below is observed. The
    # course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
    )
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)

    # The delete removes the course's link to the topic before the topic's
    # row goes: by post_delete, the course no longer covers the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Course's group as committed: the deleted topic's text is gone, only
    # the Course's own title and the other topic's text are left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.course:{basics.pk}": [
                NormalizedDocument(
                    text="Woodworking basics\n\nCarving\n\nKnives and gouges.",
                    source_app_label="testapp",
                    source_model="course",
                    source_pk=basics.pk,
                    title="Woodworking basics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_deleting_a_course_followed_by_reverse_many_to_many_replaces_the_topics_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the course's delete below is observed. The
    # topic is covered by two courses, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    joinery = Course.objects.create(title="Joinery")
    basics.topics.add(woodworking)
    joinery.topics.add(woodworking)

    # The delete removes the course's link to the topic before the course's
    # row goes: by post_delete, the topic no longer has the course.
    with django_capture_on_commit_callbacks(execute=True):
        basics.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Topic's group as committed: the deleted course's title is gone, only
    # the Topic's own title and summary and the other course's title are left.
    assert _replaced(built_outputs) == [
        {
            f"testapp.topic:{woodworking.pk}": [
                NormalizedDocument(
                    text="Woodworking\n\nJoints and finishes.\n\nJoinery",
                    source_app_label="testapp",
                    source_model="topic",
                    source_pk=woodworking.pk,
                    title="Woodworking",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_tag_of_a_photo_following_its_generic_relation_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the kind of relation can keep the saves
    # from sending anything.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Photo is registered, following its tags by the generic relation
    # ``tags``: Tag itself is not.
    rag.register(Photo, follow=["tags"])

    # Created outside the captured callbacks: the commit callback of the
    # Photo's own save never runs, so only the tag's saves below are observed.
    workbench = Photo.objects.create(title="Workbench")

    # The commit callbacks run, and an error escaping them would fail the test.
    with django_capture_on_commit_callbacks(execute=True):
        tag = Tag.objects.create(label="oak", content_object=workbench)
        tag.label = "white oak"
        tag.save()

    # A generic relation is not followed by the signals: neither the tag's
    # creation nor its update sends anything for the Photo, whose documents
    # are left stale.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_saving_a_seminar_of_a_venue_following_by_multi_column_replaces_the_venue(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_venues_following_their_seminars()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the seminar's save below is observed.
    hall = Venue.objects.create(city="Lyon", name="Halle Tony Garnier")
    # Another venue of the same city, with a seminar of its own: only both
    # columns together name a venue, so a seminar at the hall is not one of
    # its seminars.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        Seminar.objects.create(
            title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
        )
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of the venue the seminar's two columns name, with the seminar's
    # text fields, title first, after the Venue's own name; not the group of
    # the other venue of the same city.
    assert _replaced(built_outputs) == [
        {
            f"testapp.venue:{hall.pk}": [
                NormalizedDocument(
                    text=(
                        "Halle Tony Garnier\n\nAcoustics\n\nLyon\n\nHalle Tony Garnier"
                    ),
                    source_app_label="testapp",
                    source_model="venue",
                    source_pk=hall.pk,
                    title="Halle Tony Garnier",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_deleting_a_seminar_of_a_venue_following_by_multi_column_replaces_the_venue(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_venues_following_their_seminars()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the seminar's delete below is observed.
    hall = Venue.objects.create(city="Lyon", name="Halle Tony Garnier")
    deleted_seminar = Seminar.objects.create(
        title="Stage lighting", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )
    Seminar.objects.create(
        title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )

    with django_capture_on_commit_callbacks(execute=True):
        deleted_seminar.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Venue's group as committed: the deleted seminar's text is gone, the
    # remaining seminar's text fields stay after the Venue's own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.venue:{hall.pk}": [
                NormalizedDocument(
                    text=(
                        "Halle Tony Garnier\n\nAcoustics\n\nLyon\n\nHalle Tony Garnier"
                    ),
                    source_app_label="testapp",
                    source_model="venue",
                    source_pk=hall.pk,
                    title="Halle Tony Garnier",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_moving_a_seminar_of_a_venue_following_by_multi_column_replaces_both_venues(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_venues_following_their_seminars()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the seminar's move below is observed.
    hall = Venue.objects.create(city="Lyon", name="Halle Tony Garnier")
    # Another venue of the same city: the move changes only the name column,
    # so the city column alone cannot tell the two venues apart.
    transbordeur = Venue.objects.create(city="Lyon", name="Transbordeur")
    moved_seminar = Seminar.objects.create(
        title="Stage lighting", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )
    Seminar.objects.create(
        title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )

    with django_capture_on_commit_callbacks(execute=True):
        moved_seminar.venue_name = "Transbordeur"
        moved_seminar.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per Venue is not what this test is about.
    received = _received_groups(built_outputs)
    # Both Venues' groups as committed: the venue it left keeps only its
    # remaining seminar's text, the venue it joined gains the moved seminar's
    # text, each after the Venue's own name.
    assert received == {
        f"testapp.venue:{hall.pk}": [
            NormalizedDocument(
                text="Halle Tony Garnier\n\nAcoustics\n\nLyon\n\nHalle Tony Garnier",
                source_app_label="testapp",
                source_model="venue",
                source_pk=hall.pk,
                title="Halle Tony Garnier",
            ),
        ],
        f"testapp.venue:{transbordeur.pk}": [
            NormalizedDocument(
                text="Transbordeur\n\nStage lighting\n\nLyon\n\nTransbordeur",
                source_app_label="testapp",
                source_model="venue",
                source_pk=transbordeur.pk,
                title="Transbordeur",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_seminar_naming_no_venue_of_a_venue_following_by_multi_column_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_venues_following_their_seminars()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the stray seminar's saves and delete below are
    # observed. A venue of the same city, with a seminar of its own: the stray
    # seminar's city column matches it, its name column does not.
    _create_the_transbordeur_and_its_seminar()

    # The commit callbacks run, and an error escaping them would fail the test.
    with django_capture_on_commit_callbacks(execute=True):
        # A ForeignObject has no database constraint: a seminar may name a
        # (city, name) pair no Venue row has.
        stray_seminar = Seminar.objects.create(
            title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
        )
        stray_seminar.title = "Acoustics of large halls"
        stray_seminar.save()
        stray_seminar.delete()

    # The stray seminar's two columns name no venue: its saves and its delete
    # send nothing, not even for the venue of the same city.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_saving_a_seminar_a_custom_extractor_depends_on_by_multi_column_replaces_venue(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Venue is registered, with a custom extractor reading its
    # seminars: depends_on names the reverse of the multi-column ForeignObject
    # ``venue``, whose saves change its documents. Seminar itself is not
    # registered.
    @rag.register_extractor(Venue, depends_on=["seminars"])
    class VenueExtractor(BaseExtractor[Venue]):
        def extract(self, instance: Venue) -> NormalizedDocument:
            titles = [seminar.title for seminar in instance.seminars.order_by("pk")]
            return self.build_document(
                instance, text="\n\n".join([instance.name, *titles])
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the seminar's save below is observed.
    hall = Venue.objects.create(city="Lyon", name="Halle Tony Garnier")
    # Another venue of the same city, with a seminar of its own: only both
    # columns together name a venue, so a seminar at the hall is not one of
    # its seminars.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        Seminar.objects.create(
            title="Acoustics", venue_city="Lyon", venue_name="Halle Tony Garnier"
        )
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the venue the seminar's two columns name,
    # with the seminar's title after the venue's name; no group of the other
    # venue of the same city.
    assert _received_groups(built_outputs) == {
        f"testapp.venue:{hall.pk}": [
            NormalizedDocument(
                text="Halle Tony Garnier\n\nAcoustics",
                source_app_label="testapp",
                source_model="venue",
                source_pk=hall.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_venue_read_through_a_multi_column_lookup_path_replaces_its_seminars(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, reading its venue's name through a lookup
    # path across the multi-column ForeignObject ``venue``: Venue itself is not.
    rag.register(Seminar, fields=["title", "venue__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    # Another venue of the same city: only both columns together name a venue,
    # so its seminar does not read the hall.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        # Both columns are the key the seminars name the venue by: the save
        # changes neither, so the hall's seminars still read it.
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the seminars at the hall: their group, with the
    # venue's name after the Seminar's own title; not the other venue's seminar.
    assert _replaced(built_outputs) == [
        {
            f"testapp.seminar:{acoustics.pk}": [
                NormalizedDocument(
                    text="Acoustics\n\nHalle Tony Garnier",
                    source_app_label="testapp",
                    source_model="seminar",
                    source_pk=acoustics.pk,
                    title="Acoustics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_venue_a_custom_extractor_depends_on_by_multi_column_replaces_seminars(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, with a custom extractor reading its
    # venue's name: depends_on names the multi-column ForeignObject ``venue``,
    # whose saves change its documents. Venue itself is not registered.
    @rag.register_extractor(Seminar, depends_on=["venue"])
    class SeminarExtractor(BaseExtractor[Seminar]):
        def extract(self, instance: Seminar) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.venue.name}: {instance.title}"
            )

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    # Another venue of the same city: only both columns together name a venue,
    # so its seminar does not depend on the hall.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        # Both columns are the key the seminars name the venue by: the save
        # changes neither, so the hall's seminars still depend on it.
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the seminar at the hall, with the venue's
    # name; no group of the other venue's seminar.
    assert _received_groups(built_outputs) == {
        f"testapp.seminar:{acoustics.pk}": [
            NormalizedDocument(
                text="Halle Tony Garnier: Acoustics",
                source_app_label="testapp",
                source_model="seminar",
                source_pk=acoustics.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_venue_followed_by_a_multi_column_relation_replaces_its_seminars(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, following its venue through the
    # multi-column ForeignObject ``venue``: Venue itself is not.
    rag.register(Seminar, fields=["title"], follow=["venue"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    # Another venue of the same city: only both columns together name a venue,
    # so its seminar does not follow the hall.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        # Both columns are the key the seminars name the venue by: the save
        # changes neither, so the hall's seminars still follow it.
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of the seminar at the hall, with the venue's text fields, name
    # first, after the Seminar's own title; not the other venue's seminar.
    assert _replaced(built_outputs) == [
        {
            f"testapp.seminar:{acoustics.pk}": [
                NormalizedDocument(
                    text="Acoustics\n\nHalle Tony Garnier\n\nLyon",
                    source_app_label="testapp",
                    source_model="seminar",
                    source_pk=acoustics.pk,
                    title="Acoustics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_a_venue_unchanged_replaces_each_of_its_following_seminars_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, following its venue through the
    # multi-column ForeignObject ``venue``: Venue itself is not.
    rag.register(Seminar, fields=["title"], follow=["venue"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    rigging = Seminar.objects.create(
        title="Rigging", venue_city="Lyon", venue_name="Halle Tony Garnier"
    )

    with django_capture_on_commit_callbacks(execute=True):
        # The save changes neither column the seminars name the venue by: the
        # seminars that named it before the save are those that name it after.
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # One replace call holding each seminar's group once, not one batch for the
    # seminars found before the save and another for those found after it.
    assert _replaced(built_outputs) == [
        {
            f"testapp.seminar:{acoustics.pk}": [
                NormalizedDocument(
                    text="Acoustics\n\nHalle Tony Garnier\n\nLyon",
                    source_app_label="testapp",
                    source_model="seminar",
                    source_pk=acoustics.pk,
                    title="Acoustics",
                ),
            ],
            f"testapp.seminar:{rigging.pk}": [
                NormalizedDocument(
                    text="Rigging\n\nHalle Tony Garnier\n\nLyon",
                    source_app_label="testapp",
                    source_model="seminar",
                    source_pk=rigging.pk,
                    title="Rigging",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_renaming_a_venue_followed_by_multi_column_replaces_the_seminars_naming_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, following its venue through the
    # multi-column ForeignObject ``venue``: Venue itself is not.
    rag.register(Seminar, fields=["title"], follow=["venue"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's rename below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    # Another venue of the same city: the rename changes only the name column,
    # so the city column alone cannot tell its seminar from the hall's.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        # The name is one of the columns the seminars name the venue by: after
        # the save, the hall's seminar still carries the old name, so only the
        # venue as it was before the save tells which seminars named it.
        hall.name = "Grande Halle"
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of the seminar that named the hall, as committed: its two
    # columns now name no venue, so only the Seminar's own title is left, not
    # the stale venue text; no group of the other venue's seminar.
    assert _replaced(built_outputs) == [
        {
            f"testapp.seminar:{acoustics.pk}": [
                NormalizedDocument(
                    text="Acoustics",
                    source_app_label="testapp",
                    source_model="seminar",
                    source_pk=acoustics.pk,
                    title="Acoustics",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_renaming_a_venue_a_custom_extractor_depends_on_replaces_the_seminars_naming_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, with a custom extractor reading its
    # venue's name when its two columns name one: depends_on names the
    # multi-column ForeignObject ``venue``, whose saves change its documents.
    # Venue itself is not registered.
    @rag.register_extractor(Seminar, depends_on=["venue"])
    class SeminarExtractor(BaseExtractor[Seminar]):
        def extract(self, instance: Seminar) -> NormalizedDocument:
            try:
                venue = instance.venue
            except Venue.DoesNotExist:
                return self.build_document(instance, text=instance.title)
            return self.build_document(instance, text=f"{venue.name}: {instance.title}")

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's rename below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    # Another venue of the same city: the rename changes only the name column,
    # so the city column alone cannot tell its seminar from the hall's.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        # The name is one of the columns the seminars name the venue by: after
        # the save, the hall's seminar still carries the old name, so only the
        # venue as it was before the save tells which seminars named it.
        hall.name = "Grande Halle"
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The group of the seminar that named the hall, as
    # committed: its two columns now name no venue, so only its title is
    # left, not the stale venue name; no group of the other venue's seminar.
    assert _received_groups(built_outputs) == {
        f"testapp.seminar:{acoustics.pk}": [
            NormalizedDocument(
                text="Acoustics",
                source_app_label="testapp",
                source_model="seminar",
                source_pk=acoustics.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_saving_a_venue_read_past_a_foreign_key_then_multi_column_replaces_its_talks(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Talk is registered, reading its seminar's venue's name through a
    # lookup path whose later link is the multi-column ForeignObject ``venue``:
    # neither Seminar nor Venue is.
    rag.register(Talk, fields=["title", "seminar__venue__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    keynote = Talk.objects.create(title="Keynote", seminar=acoustics)
    # Another venue of the same city: only both columns together name a venue,
    # so the talk of its seminar does not read the hall.
    lighting = _create_the_transbordeur_and_its_seminar()
    Talk.objects.create(title="Spotlights", seminar=lighting)

    with django_capture_on_commit_callbacks(execute=True):
        # Both columns are the key the seminars name the venue by: the save
        # changes neither, so the talks of the hall's seminars still read it.
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The path alone resyncs the talks of the seminars at the hall: their
    # group, with the venue's name after the Talk's own title; not the talk at
    # the other venue.
    assert _replaced(built_outputs) == [
        {
            f"testapp.talk:{keynote.pk}": [
                NormalizedDocument(
                    text="Keynote\n\nHalle Tony Garnier",
                    source_app_label="testapp",
                    source_model="talk",
                    source_pk=keynote.pk,
                    title="Keynote",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_renaming_a_venue_read_past_a_foreign_key_then_multi_column_replaces_its_talks(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Talk is registered, reading its seminar's venue's name through a
    # lookup path whose later link is the multi-column ForeignObject ``venue``:
    # neither Seminar nor Venue is.
    rag.register(Talk, fields=["title", "seminar__venue__name"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's rename below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    keynote = Talk.objects.create(title="Keynote", seminar=acoustics)
    # Another venue of the same city: the rename changes only the name column,
    # so the city column alone cannot tell the talk of its seminar from the
    # talk of the hall's.
    lighting = _create_the_transbordeur_and_its_seminar()
    Talk.objects.create(title="Spotlights", seminar=lighting)

    with django_capture_on_commit_callbacks(execute=True):
        # The name is one of the columns the seminars name the venue by: after
        # the save, the hall's seminar still carries the old name, so only the
        # venue as it was before the save tells which talks read it.
        hall.name = "Grande Halle"
        hall.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The group of the talk of the seminar that named the hall, as committed:
    # its seminar's two columns now name no venue, so only the Talk's own title
    # is left, not the stale venue name; no group of the other venue's talk.
    assert _replaced(built_outputs) == [
        {
            f"testapp.talk:{keynote.pk}": [
                NormalizedDocument(
                    text="Keynote",
                    source_app_label="testapp",
                    source_model="talk",
                    source_pk=keynote.pk,
                    title="Keynote",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_a_save_sends_the_documents_of_the_instance_as_committed(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        # A queryset update sends no signal and leaves the saved Python object
        # as it was: only the committed row carries the new name.
        Category.objects.filter(pk=lighting.pk).update(name="Lamps")

    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{lighting.pk}": [
                NormalizedDocument(
                    text="Lamps",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=lighting.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_saving_an_instance_with_no_document_replaces_its_group_with_an_empty_one(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> None:
            # Skipped: the instance produces no document.
            return None

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        assert _replaced(built_outputs) == []

    # An empty group, so the output drops what it held for the instance.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting.pk}": []}]


@pytest.mark.django_db
def test_an_instance_saved_then_deleted_before_the_commit_sends_an_empty_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        lighting_pk = lighting.pk
        # The row is gone by the commit: the save has nothing left to extract.
        lighting.delete()

    # An empty group, so the output holds nothing for the deleted instance.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting_pk}": []}]


@pytest.mark.django_db
def test_deleting_a_registered_instance_replaces_its_group_with_an_empty_one(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Created in a commit of its own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    lighting_pk = lighting.pk
    built_outputs.clear()

    with django_capture_on_commit_callbacks(execute=True):
        lighting.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # An empty group, so the output drops what it held for the instance.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting_pk}": []}]


@pytest.mark.django_db
def test_deleting_a_queryset_empties_the_group_of_each_deleted_instance(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Created in a commit of their own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        lamps = Category.objects.create(name="Lamps")
        tools = Category.objects.create(name="Tools")
    built_outputs.clear()

    with django_capture_on_commit_callbacks(execute=True):
        # One queryset delete for several instances; Tools is kept.
        Category.objects.exclude(pk=tools.pk).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: whether the groups come in one call or one
    # per instance is not what this test is about.
    received = _received_groups(built_outputs)
    # An empty group per deleted instance, and nothing for the kept one.
    assert received == {
        f"testapp.category:{lighting.pk}": [],
        f"testapp.category:{lamps.pk}": [],
    }


@pytest.mark.django_db
def test_deleting_through_a_proxy_empties_the_group_of_the_registered_model(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the concrete model is registered, not its proxy.
    _register_categories_by_name()

    # Created in a commit of its own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    lighting_pk = lighting.pk
    built_outputs.clear()

    # Django sends post_delete with the proxy as its sender, not Category.
    with django_capture_on_commit_callbacks(execute=True):
        CategoryProxy.objects.get(pk=lighting_pk).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The empty group is the registered model's, under its label, not the
    # proxy's.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting_pk}": []}]


@pytest.mark.django_db
def test_unregistering_a_proxy_keeps_deleting_through_it_emptying_the_models_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The concrete model and its proxy are both registered, then the proxy
    # alone is unregistered: Category stays registered.
    _register_categories_by_name()
    rag.register(CategoryProxy, fields=["name"])
    rag.unregister(CategoryProxy)

    # Created in a commit of its own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    lighting_pk = lighting.pk
    built_outputs.clear()

    # Django sends post_delete with the proxy as its sender, not Category.
    with django_capture_on_commit_callbacks(execute=True):
        CategoryProxy.objects.get(pk=lighting_pk).delete()

    # Category's registration still listens to its proxy's deletions.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting_pk}": []}]


@pytest.mark.django_db
def test_deleting_a_multi_table_child_empties_the_group_of_its_registered_parent(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the parent is registered, not its multi-table child.
    _register_products_by_name()

    # Created in a commit of their own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        lamp = _create_a_desk_lamp(lighting)
    lamp_pk = lamp.pk
    built_outputs.clear()

    # Deleting the child deletes its parent row too.
    with django_capture_on_commit_callbacks(execute=True):
        lamp.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The empty group is the parent row's, under the parent's label.
    assert _replaced(built_outputs) == [{f"testapp.product:{lamp_pk}": []}]


@pytest.mark.django_db
def test_deleting_a_child_with_a_primary_key_of_its_own_empties_its_parents_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the parent is registered, not its multi-table child.
    _register_products_by_name()

    # Created in a commit of their own, so that only the delete's commit is
    # observed below.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        lamp = ClearanceProduct.objects.create(
            code="CLR-1",
            name="Desk lamp",
            description="A lamp for the desk.",
            price="25.00",
            category=lighting,
        )
    # The child's primary key is its code, not the Product's: the parent row
    # is reached by the explicit parent link, ``product``.
    product_pk = lamp.product_id
    built_outputs.clear()

    # Deleting the child deletes its parent row too: Django sends post_delete
    # for each.
    with django_capture_on_commit_callbacks(execute=True):
        lamp.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Exactly one empty group, the parent row's, under the parent's label and
    # the parent's primary key, not the child's code.
    assert _replaced(built_outputs) == [{f"testapp.product:{product_pk}": []}]


@pytest.mark.django_db
def test_deleting_a_followed_related_instance_replaces_the_group_that_follows_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the plugin's delete below is observed.
    page = Page.objects.create(title="About us", slug="about-us")
    deleted_plugin = TextPlugin.objects.create(
        page=page, body="We build chairs by hand."
    )
    TextPlugin.objects.create(page=page, body="We ship worldwide.")

    with django_capture_on_commit_callbacks(execute=True):
        deleted_plugin.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Page's group as committed: the deleted plugin's text is gone, the
    # remaining plugin's text stays after the Page's own title.
    assert _replaced(built_outputs) == [
        {
            f"testapp.page:{page.pk}": [
                NormalizedDocument(
                    text="About us\n\nWe ship worldwide.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=page.pk,
                    title="About us",
                    url="/pages/about-us/",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_deleting_a_page_whose_followed_plugins_cascade_sends_only_its_empty_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the Page's delete below is observed.
    page = Page.objects.create(title="About us", slug="about-us")
    page_pk = page.pk
    TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    TextPlugin.objects.create(page=page, body="We ship worldwide.")

    # The plugins are deleted with the Page by cascade: Django sends
    # post_delete for each of them as well as for the Page.
    with django_capture_on_commit_callbacks(execute=True):
        page.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Page's empty group, once, and no replacement of it with documents.
    assert _replaced(built_outputs) == [{f"testapp.page:{page_pk}": []}]


@pytest.mark.django_db
def test_saving_an_instance_of_an_unregistered_model_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Another model is registered, so the registry is not simply empty.
    _register_products_by_name()

    with django_capture_on_commit_callbacks(execute=True):
        Category.objects.create(name="Lighting")

    # Not even an output built: nothing reaches the backend.
    assert built_outputs == []


def test_an_unregistered_model_keeps_the_fast_delete_of_django() -> None:
    # Another model is registered, so the registry is not simply empty.
    _register_products_by_name()

    # Django's deletion Collector fast-deletes (one DELETE query, no instance
    # loaded) only a model with no post_delete listener; a receiver connected
    # without a sender listens to every model.
    assert not post_delete.has_listeners(Category)


@pytest.mark.django_db
def test_saving_an_instance_of_a_model_since_unregistered_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    rag.unregister(Category)

    with django_capture_on_commit_callbacks(execute=True):
        Category.objects.create(name="Lighting")

    # Not even an output built: the registration it once had is forgotten.
    assert built_outputs == []


@pytest.mark.django_db
def test_a_plugin_of_a_page_since_unregistered_defers_nothing_and_fast_deletes(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the unregistration can keep the save and
    # the delete from deferring anything to the commit.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    # The same instance is updated once while the Page is still registered:
    # its save sends the Page's group.
    with django_capture_on_commit_callbacks(execute=True):
        plugin.body = "We ship worldwide."
        plugin.save()
    replaced_while_registered = _replaced(built_outputs)
    # Merged across replace calls: how many calls the update makes is not what
    # this test is about.
    assert list(_received_groups(built_outputs)) == [f"testapp.page:{page.pk}"]

    rag.unregister(Page)

    # Django's deletion Collector fast-deletes a model with no post_delete
    # listener: nothing is left listening to TextPlugin's deletes.
    assert not post_delete.has_listeners(TextPlugin)

    # The commit callbacks run: one sending anything would reach the output.
    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        plugin.body = "We build tables by hand."
        plugin.save()
        plugin.delete()

    # The registration the Page once had is forgotten: nothing is deferred to
    # the commit, and nothing more reaches the output.
    assert callbacks == []
    assert _replaced(built_outputs) == replaced_while_registered


def test_a_forward_followed_topic_is_listened_to_only_while_followed() -> None:
    # Two registered models follow Topic through their own foreign key;
    # neither Topic nor its proxy is registered.
    rag.register(Workshop, follow=["topic"])
    rag.register(Lesson, follow=["topic"])

    rag.unregister(Workshop)

    # Lesson still follows Topic: its deletes, through Topic or its proxy,
    # are still listened to.
    for sender in (Topic, TopicProxy):
        assert pre_delete.has_listeners(sender)
        assert post_delete.has_listeners(sender)

    rag.unregister(Lesson)

    # Django's deletion Collector fast-deletes only a model with no pre_delete
    # and no post_delete listener: once no registered model follows Topic,
    # nothing is left listening to its deletes, through Topic or its proxy.
    for sender in (Topic, TopicProxy):
        assert not pre_delete.has_listeners(sender)
        assert not post_delete.has_listeners(sender)


def test_a_topic_custom_extractors_depend_on_is_listened_to_only_while_needed() -> None:
    # Two registered models have a custom extractor depending on Topic through
    # their own foreign key; neither Topic nor its proxy is registered.
    @rag.register_extractor(Workshop, depends_on=["topic"])
    class WorkshopExtractor(BaseExtractor[Workshop]):
        def extract(self, instance: Workshop) -> NormalizedDocument:
            return self.build_document(instance, text=instance.title)

    @rag.register_extractor(Lesson, depends_on=["topic"])
    class LessonExtractor(BaseExtractor[Lesson]):
        def extract(self, instance: Lesson) -> NormalizedDocument:
            return self.build_document(instance, text=instance.title)

    rag.unregister(Workshop)

    # Lesson still depends on Topic: its deletes, through Topic or its proxy,
    # are still listened to.
    for sender in (Topic, TopicProxy):
        assert pre_delete.has_listeners(sender)
        assert post_delete.has_listeners(sender)

    rag.unregister(Lesson)

    # Django's deletion Collector fast-deletes only a model with no pre_delete
    # and no post_delete listener: once no registered model depends on Topic,
    # nothing is left listening to its deletes, through Topic or its proxy.
    for sender in (Topic, TopicProxy):
        assert not pre_delete.has_listeners(sender)
        assert not post_delete.has_listeners(sender)


@pytest.mark.django_db
def test_each_commit_builds_a_new_output_with_the_configured_options(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": TRACKED_BACKEND,
        "OPTIONS": {"collection": "catalog", "batch_size": 50},
    }

    _register_categories_by_name()

    # Two separate transactions, each with its own commit.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
    with django_capture_on_commit_callbacks(execute=True):
        tools = Category.objects.create(name="Tools")

    [first_output, second_output] = built_outputs
    assert first_output is not second_output
    assert first_output.options == {"collection": "catalog", "batch_size": 50}
    assert second_output.options == {"collection": "catalog", "batch_size": 50}
    # Each output receives the group of its own commit's save, and only it.
    assert [list(groups) for groups in first_output.replaced] == [
        [f"testapp.category:{lighting.pk}"]
    ]
    assert [list(groups) for groups in second_output.replaced] == [
        [f"testapp.category:{tools.pk}"]
    ]


@pytest.mark.django_db
def test_saving_a_registered_instance_without_an_output_setting_fails_at_the_save(
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    _register_categories_by_name()

    # The commit callbacks are captured and never run: only an error raised by
    # the save itself is caught, not one deferred to the commit.
    with (
        django_capture_on_commit_callbacks(execute=False),
        pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"),
    ):
        Category.objects.create(name="Lighting")


@pytest.mark.django_db(transaction=True)
def test_a_save_in_autocommit_with_no_output_setting_fails_and_writes_no_row() -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    _register_categories_by_name()

    # No transaction around the save (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the save must fail before its
    # INSERT does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        Category.objects.create(name="Lighting")

    assert not Category.objects.exists()


@pytest.mark.django_db(transaction=True)
def test_a_followed_instance_saved_with_no_output_setting_fails_and_writes_no_row() -> (
    None
):
    # Created before Page is registered: with no MODEL_RAG_OUTPUT, its own
    # save would fail otherwise.
    page = Page.objects.create(title="About us", slug="about-us")

    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    _register_pages_following_their_plugins()

    # No transaction around the save (transaction=True): in autocommit, each
    # query commits as soon as it runs, so the save must fail before its
    # INSERT does.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    assert not TextPlugin.objects.exists()


@pytest.mark.django_db
def test_saving_a_registered_instance_with_a_backend_lacking_replace_fails_at_the_save(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A callable prune, but no replace at all.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": "tests.recording.PruneOnlyOutput"}

    _register_categories_by_name()

    # The commit callbacks are captured and never run: only an error raised by
    # the save itself is caught, not one deferred to the commit.
    with (
        django_capture_on_commit_callbacks(execute=False),
        pytest.raises(ImproperlyConfigured, match="PruneOnlyOutput.*replace"),
    ):
        Category.objects.create(name="Lighting")


@pytest.mark.django_db
def test_saving_a_registered_instance_with_options_the_backend_rejects_fails_at_save(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # RecordingOutput is built with no argument at all: it accepts no option.
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": "tests.recording.RecordingOutput",
        "OPTIONS": {"collection": "catalog"},
    }

    _register_categories_by_name()

    # The commit callbacks are captured and never run: only an error raised by
    # the save itself is caught, not one deferred to the commit.
    with (
        django_capture_on_commit_callbacks(execute=False),
        pytest.raises(ImproperlyConfigured, match="RecordingOutput.*collection"),
    ):
        Category.objects.create(name="Lighting")


@pytest.mark.django_db
def test_saving_with_a_backend_whose_signature_is_unreadable_replaces_its_group(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": TRACKED_BACKEND,
        "OPTIONS": {"collection": "catalog"},
    }
    real_signature = inspect.signature

    # Some classes, such as those implemented in C, have no signature Python
    # can read: inspect.signature raises ValueError for them. Simulated here
    # for the tracked backend only, which can still be built with its options.
    def signature_unreadable_for_the_backend(
        obj: Any, *args: Any, **kwargs: Any
    ) -> inspect.Signature:
        if obj is TrackedRecordingOutput:
            message = f"no signature found for {obj!r}"
            raise ValueError(message)
        return real_signature(obj, *args, **kwargs)

    monkeypatch.setattr(inspect, "signature", signature_unreadable_for_the_backend)

    _register_categories_by_name()

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")

    [output] = built_outputs
    assert output.options == {"collection": "catalog"}
    assert _received_groups(built_outputs) == {
        f"testapp.category:{lighting.pk}": [
            NormalizedDocument(
                text="Lighting",
                source_app_label="testapp",
                source_model="category",
                source_pk=lighting.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_deleting_a_registered_instance_without_an_output_setting_fails_at_the_delete(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # Configured only while the instance is created, so the save succeeds.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    with django_capture_on_commit_callbacks(execute=False):
        lighting = Category.objects.create(name="Lighting")

    # Back to tests/settings.py, which defines no MODEL_RAG_OUTPUT.
    del settings.MODEL_RAG_OUTPUT

    # The commit callbacks are captured and never run: only an error raised by
    # the delete itself is caught, not one deferred to the commit.
    with (
        django_capture_on_commit_callbacks(execute=False),
        pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"),
    ):
        lighting.delete()


@pytest.mark.django_db
def test_deleting_a_followed_instance_without_an_output_setting_fails_at_the_delete(
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # Created before Page is registered: with no MODEL_RAG_OUTPUT, their own
    # saves would fail otherwise.
    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    _register_pages_following_their_plugins()

    # The commit callbacks are captured and never run: only an error raised by
    # the delete itself is caught, not one deferred to the commit.
    with (
        django_capture_on_commit_callbacks(execute=False),
        pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"),
    ):
        plugin.delete()


@pytest.mark.django_db
def test_saving_a_registered_instance_with_signals_off_sends_nothing_and_raises_nothing(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_SIGNALS = False
    # tests/settings.py defines no MODEL_RAG_OUTPUT: with signals on, the save
    # itself would raise ImproperlyConfigured.

    _register_categories_by_name()

    # The commit callbacks run: one sending anything would need an output and
    # raise for lack of one.
    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        Category.objects.create(name="Lighting")

    # Nothing is even deferred to the commit.
    assert callbacks == []


@pytest.mark.django_db
def test_deleting_a_registered_instance_with_signals_off_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_SIGNALS = False
    # A working output, so that only the setting can keep the delete from
    # sending to it.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    lighting = Category.objects.create(name="Lighting")

    # The commit callbacks run: one sending anything would build an output.
    with django_capture_on_commit_callbacks(execute=True) as callbacks:
        lighting.delete()

    # Nothing is even deferred to the commit, and no output is built.
    assert callbacks == []
    assert built_outputs == []


@pytest.mark.django_db
def test_saving_and_deleting_a_followed_instance_with_signals_off_costs_nothing_more(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_SIGNALS = False
    # A working output, so that only the setting can keep the save and the
    # delete from deferring anything to the commit.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    # The commit callbacks run: one sending anything would build an output.
    # With signals on, the save of an existing row would read its committed
    # followers first: one query more than the save's own.
    with (
        django_capture_on_commit_callbacks(execute=True) as callbacks,
        django_assert_num_queries(2) as queries,
    ):
        plugin.body = "We ship worldwide."
        plugin.save()
        plugin.delete()

    # Nothing is deferred to the commit, and the only queries are the save's
    # UPDATE and the delete's DELETE.
    assert callbacks == []
    statements = _statements(queries)
    assert statements == ["UPDATE", "DELETE"]


@pytest.mark.django_db
def test_a_raw_save_of_a_registered_instance_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the raw save can keep it from being sent to.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Saved as loaddata saves a fixture: a deserialized object's save() is a
    # raw save, so Django sends post_save with raw=True.
    fixture = [{"model": "testapp.category", "pk": 1, "fields": {"name": "Lighting"}}]
    with django_capture_on_commit_callbacks(execute=True):
        for deserialized in serializers.deserialize("python", fixture):
            deserialized.save()

    # The row is there, yet not even an output is built, even after the commit.
    assert Category.objects.filter(name="Lighting").exists()
    assert built_outputs == []


@pytest.mark.django_db
def test_a_raw_save_of_a_followed_instance_defers_nothing_and_reads_nothing_first(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the raw saves can keep them from deferring
    # anything to the commit.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    page = Page.objects.create(title="About us", slug="about-us")
    plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    plugin = TextPlugin.objects.get(pk=plugin.pk)

    # A new row saved as loaddata saves a fixture, then an existing row saved
    # the way loaddata saves each deserialized object: save_base(raw=True).
    # With a regular save, the existing row would read its committed followers
    # first.
    fixture = [
        {
            "model": "testapp.textplugin",
            "pk": plugin.pk + 1,
            "fields": {"page": page.pk, "body": "We ship worldwide."},
        }
    ]
    with (
        django_capture_on_commit_callbacks(execute=True) as callbacks,
        CaptureQueriesContext(connection) as queries,
    ):
        for deserialized in serializers.deserialize("python", fixture):
            deserialized.save()
        plugin.body = "We build tables by hand."
        plugin.save_base(raw=True)

    # Both rows are written, yet nothing is deferred to the commit, no output
    # is built, and the only queries are the saves' writes: none reads anything.
    assert TextPlugin.objects.filter(page=page).count() == len(fixture) + 1
    assert callbacks == []
    assert built_outputs == []
    statements = set(_statements(queries))
    assert statements <= {"UPDATE", "INSERT"}


class _RolledBackError(Exception):
    """Raised inside an atomic block to roll its transaction back."""


@pytest.mark.django_db(transaction=True)
def test_saving_a_registered_instance_in_a_rolled_back_transaction_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # A real transaction (transaction=True), so that leaving the atomic block
    # on an exception rolls it back rather than a savepoint of the test's own.
    with pytest.raises(_RolledBackError), transaction.atomic():
        Category.objects.create(name="Lighting")
        raise _RolledBackError

    assert _replaced(built_outputs) == []


class _ExtractionError(Exception):
    """Raised by an extractor that fails on the instance it is given."""


@pytest.mark.django_db
def test_an_extractor_failing_at_the_commit_of_a_save_is_logged_without_raising(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            raise _ExtractionError

    # The commit callbacks run, and an error escaping them would fail the test:
    # the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting = Category.objects.create(name="Lighting")

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record says which instance failed, and carries the error itself.
    assert f"testapp.category:{lighting.pk}" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], _ExtractionError)
    # Not even an empty group: what the output held for the instance is kept.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_a_follower_failing_at_the_commit_of_a_followed_save_is_logged_without_raising(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callback of the
    # Page's own save never runs, so only the plugin's save below is observed.
    page = Page.objects.create(title="About us", slug="about-us")

    def fail_to_extract(self: object, instance: object) -> NormalizedDocument:
        raise _ExtractionError

    # The Page's extractor, the one rag.register built, fails on every Page.
    monkeypatch.setattr(type(rag.new_extractor(Page)), "extract", fail_to_extract)

    # The commit callbacks run, and an error escaping them would fail the test:
    # the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        plugin = TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the followers' model and the plugin saved, and carries
    # the error.
    assert record.getMessage() == (
        f"Syncing testapp.page instances that follow testapp.textplugin:{plugin.pk} "
        "failed"
    )
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], _ExtractionError)
    # Not even an empty group: what the output held for the Page is kept.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_deleting_an_instance_empties_its_group_without_going_through_its_extractor(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Created before Category is registered, so that its save schedules
    # nothing: only the delete's commit is observed below.
    lighting = Category.objects.create(name="Lighting")
    lighting_pk = lighting.pk

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def get_queryset(self, queryset: QuerySet[Category]) -> QuerySet[Category]:
            raise _ExtractionError

        def extract(self, instance: Category) -> NormalizedDocument:
            raise _ExtractionError

    # An error escaping the commit callbacks would fail the test: the commit
    # itself must not raise.
    with django_capture_on_commit_callbacks(execute=True):
        lighting.delete()

    # The row is gone, so there is nothing for the extractor to filter or
    # extract: the empty group is sent all the same.
    assert _replaced(built_outputs) == [{f"testapp.category:{lighting_pk}": []}]


@pytest.mark.django_db
def test_an_output_failing_on_one_saved_instance_still_receives_the_other_ones_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Created in a commit of their own, so that their source keys are known
    # before the output is configured to fail on one of them.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        tools = Category.objects.create(name="Tools")
    built_outputs.clear()

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.category:{lighting.pk}"},
    }

    # The failing instance is saved first, so that its failure comes before the
    # other instance's group is sent. An error escaping the commit callbacks
    # would fail the test: the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting.name = "Lamps"
        lighting.save()
        tools.name = "Hardware"
        tools.save()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record says which instance failed, and carries the error itself.
    assert f"testapp.category:{lighting.pk}" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], FailingReplaceError)
    # The other instance's group still reaches an output.
    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{tools.pk}": [
                NormalizedDocument(
                    text="Hardware",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=tools.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_an_output_failing_on_one_deleted_instance_still_receives_the_other_ones_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Created in a commit of their own, so that their source keys are known
    # before the output is configured to fail on one of them.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        tools = Category.objects.create(name="Tools")
    lighting_pk = lighting.pk
    tools_pk = tools.pk
    built_outputs.clear()

    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.category:{lighting_pk}"},
    }

    # Deleted one by one, so that each sends its own signal. The failing
    # instance is deleted first, so that its failure comes before the other
    # instance's empty group is sent. An error escaping the commit callbacks
    # would fail the test: the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting.delete()
        tools.delete()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record says which instance failed, and carries the error itself.
    assert f"testapp.category:{lighting_pk}" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], FailingReplaceError)
    # The other instance's empty group still reaches an output.
    assert _replaced(built_outputs) == [{f"testapp.category:{tools_pk}": []}]


def _selects_by(sql: str, params: Any, lookup: str, value: Any) -> bool:
    """Whether ``sql`` is a SELECT whose ``lookup`` compares a column with ``value``.

    The parameter compared is the lookup's own, found by counting the
    placeholders before it: a query's other parameters (such as the constant
    exists() selects) may hold the same value without being the lookup's.
    """
    if not sql.startswith("SELECT") or lookup not in sql:
        return False
    lookup_param_index = sql[: sql.index(lookup)].count("%s")
    return bool(params[lookup_param_index] == value)


# How a query on Category compares a row's primary key with a parameter.
CATEGORY_PK_LOOKUP = '"testapp_category"."id" = %s'


def _reads_category_row(sql: str, params: Any, pk: Any) -> bool:
    """Whether ``sql`` is a SELECT looking up the Category row of primary key ``pk``."""
    return _selects_by(sql, params, CATEGORY_PK_LOOKUP, pk)


@pytest.mark.django_db
def test_a_database_error_reloading_a_saved_instance_is_logged_without_raising(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_categories_by_name()

    # Created in a commit of their own, so that the database can be made to
    # fail on reading one of them back.
    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        tools = Category.objects.create(name="Tools")
    built_outputs.clear()

    reload_error = DatabaseError("the database is unreachable")

    def fail_reading_lighting(
        execute: Callable[[str, Any, bool, dict[str, Any]], Any],
        sql: str,
        params: Any,
        many: bool,
        context: dict[str, Any],
    ) -> Any:
        # Only reading Lighting's row fails: its save, an UPDATE, goes through,
        # and so does every query on Tools.
        if _reads_category_row(sql, params, lighting.pk):
            raise reload_error
        return execute(sql, params, many, context)

    # The failing instance is saved first, so that its failure comes before the
    # other instance's group is sent. An error escaping the commit callbacks
    # would fail the test: the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        connection.execute_wrapper(fail_reading_lighting),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting.name = "Lamps"
        lighting.save()
        tools.name = "Hardware"
        tools.save()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record says which instance failed, and carries the error itself.
    assert f"testapp.category:{lighting.pk}" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is reload_error
    # The other instance's group still reaches an output.
    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{tools.pk}": [
                NormalizedDocument(
                    text="Hardware",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=tools.pk,
                ),
            ],
        }
    ]


# How a query on Product compares a row's category with a parameter.
PRODUCT_CATEGORY_LOOKUP = '"testapp_product"."category_id" = %s'


def _looks_up_products_of_category(sql: str, params: Any, category_pk: Any) -> bool:
    """Whether ``sql`` is a SELECT looking up the Products of the Category of
    primary key ``category_pk``."""
    return _selects_by(sql, params, PRODUCT_CATEGORY_LOOKUP, category_pk)


@pytest.mark.django_db
def test_a_database_error_looking_up_a_categorys_followers_at_the_commit_is_logged(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    rag.register(Product, fields=["name"], follow=["category"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the categories' saves below are observed.
    lighting = Category.objects.create(name="Lighting")
    desk_lamp = _create_a_plain_desk_lamp(lighting)
    tools = Category.objects.create(name="Tools")
    hammer = _create_a_hammer(tools)

    # The saves themselves go through: their callbacks are captured here, and
    # run below as the commit would run them, once the database fails. The
    # followers each category had before its save are found by then.
    with django_capture_on_commit_callbacks() as callbacks:
        lighting.name = "Lamps"
        lighting.save()
        tools.name = "Hardware"
        tools.save()

    lookup_error = DatabaseError("the database is unreachable")

    def fail_looking_up_lighting_followers(
        execute: Callable[[str, Any, bool, dict[str, Any]], Any],
        sql: str,
        params: Any,
        many: bool,
        context: dict[str, Any],
    ) -> Any:
        # Only looking up the Products of Lighting fails: every query on the
        # Products of Tools goes through.
        if _looks_up_products_of_category(sql, params, lighting.pk):
            raise lookup_error
        return execute(sql, params, many, context)

    # The callbacks run in order, the failing category's first, so that its
    # failure comes before the other category's followers are sent. An error
    # escaping a callback would fail the test: the commit itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        connection.execute_wrapper(fail_looking_up_lighting_followers),
    ):
        for callback in callbacks:
            callback()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the category saved, and carries the error itself.
    assert f"testapp.category:{lighting.pk}" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is lookup_error
    # The follower the failing category had before its save, found then, still
    # reaches an output, and so does the other category's follower.
    assert _replaced(built_outputs) == [
        {
            f"testapp.product:{desk_lamp.pk}": [
                NormalizedDocument(
                    text="Desk lamp\n\nLamps",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=desk_lamp.pk,
                    title="Desk lamp",
                    url=f"/products/{desk_lamp.pk}/",
                ),
            ],
        },
        {
            f"testapp.product:{hammer.pk}": [
                NormalizedDocument(
                    text="Hammer\n\nHardware",
                    source_app_label="testapp",
                    source_model="product",
                    source_pk=hammer.pk,
                    title="Hammer",
                    url=f"/products/{hammer.pk}/",
                ),
            ],
        },
    ]
