import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError, connection, transaction
from django.db.models import QuerySet
from django.db.models.signals import post_delete
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
    Category,
    CategoryProxy,
    ClearanceProduct,
    Course,
    Exhibit,
    FeaturedProduct,
    Page,
    Product,
    Shelf,
    Showroom,
    Supplier,
    SupplierProfile,
    TextPlugin,
    TextPluginProxy,
    Topic,
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


def _create_a_desk_lamp(category: Category) -> FeaturedProduct:
    """Create a Desk lamp, a FeaturedProduct: a Product row and its child row."""
    return FeaturedProduct.objects.create(
        name="Desk lamp",
        description="A lamp for the desk.",
        price="25.00",
        category=category,
        tagline="Light up your work",
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
def test_a_course_of_a_topic_following_a_reverse_many_to_many_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the kind of relation can keep the save
    # and the delete from sending anything.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse
    # many-to-many ``courses``: Course itself is not.
    rag.register(Topic, follow=["courses"])

    # Created outside the captured callbacks: the commit callback of the
    # Topic's own save never runs, so only the course's saves and delete below
    # are observed.
    joinery = Topic.objects.create(
        summary="Joining wood.", title="Joinery", slug="joinery"
    )

    # The commit callbacks run, and an error escaping them would fail the test.
    with django_capture_on_commit_callbacks(execute=True):
        course = Course.objects.create(title="Woodworking basics")
        course.topics.add(joinery)
        course.title = "Woodworking for beginners"
        course.save()
        course.delete()

    # A Course row holds no key of a Topic: its saves and deletes send nothing
    # for the Topic.
    assert _replaced(built_outputs) == []
    # Django's deletion Collector fast-deletes a model with no post_delete
    # listener: nothing listens to Course's deletes.
    assert not post_delete.has_listeners(Course)


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
        TextPlugin.objects.create(page=page, body="We build chairs by hand.")

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the Page's group, not the plugin, and carries the error.
    assert f"testapp.page:{page.pk}" in record.getMessage()
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


# How a query on Category compares a row's primary key with a parameter.
CATEGORY_PK_LOOKUP = '"testapp_category"."id" = %s'


def _reads_category_row(sql: str, params: Any, pk: Any) -> bool:
    """Whether ``sql`` is a SELECT looking up the Category row of primary key ``pk``.

    The parameter compared is the lookup's own, found by counting the
    placeholders before it: a query's other parameters (such as the constant
    exists() selects) may hold the same value without reading that row.
    """
    if not sql.startswith("SELECT") or CATEGORY_PK_LOOKUP not in sql:
        return False
    lookup_param_index = sql[: sql.index(CATEGORY_PK_LOOKUP)].count("%s")
    return bool(params[lookup_param_index] == pk)


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
