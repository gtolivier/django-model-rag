import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

import pytest
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.db import DatabaseError, connection, transaction
from django.db.models import Model, QuerySet
from django.db.models.signals import m2m_changed, post_delete, pre_delete
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
    Band,
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
    MasterClass,
    Match,
    Meetup,
    Musician,
    Note,
    Notice,
    Offer,
    Page,
    PageIntro,
    Person,
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
    Team,
    TextPlugin,
    TextPluginProxy,
    Theme,
    Topic,
    TopicProxy,
    Tournament,
    Venue,
    Warehouse,
    Workshop,
)

# The logger the package reports a failed commit callback on.
PACKAGE_LOGGER = "django_model_rag"

# The replace calls an output receives, in call order.
ReplaceCalls = list[Mapping[str, Sequence[NormalizedDocument]]]


def _replaced(built_outputs: list[TrackedRecordingOutput]) -> ReplaceCalls:
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


def _register_courses_following_their_topics() -> None:
    """Register only Course, following its topics through its own many-to-many
    ``topics``: Topic itself is not."""
    rag.register(Course, follow=["topics"])


def _register_topics_following_their_courses() -> None:
    """Register only Topic, following its courses by the reverse many-to-many
    ``courses``: Course itself is not."""
    rag.register(Topic, follow=["courses"])


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


def _register_pages_by_their_plugin_bodies() -> None:
    """Register only Page, with a custom extractor reading its title, then the
    bodies of its text plugins: depends_on names the reverse relation
    ``text_plugins``, whose saves change its documents. TextPlugin itself is
    not registered."""

    @rag.register_extractor(Page, depends_on=["text_plugins"])
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> NormalizedDocument:
            bodies = [plugin.body for plugin in instance.text_plugins.order_by("pk")]
            return self.build_document(
                instance, text="\n\n".join([instance.title, *bodies])
            )


def _register_topics_by_their_course_titles() -> None:
    """Register only Topic, with a custom extractor reading its title, then the
    titles of its courses: depends_on names its reverse many-to-many
    ``courses``, whose saves, links and deletes change its documents. Course
    itself is not registered."""

    @rag.register_extractor(Topic, depends_on=["courses"])
    class TopicExtractor(BaseExtractor[Topic]):
        def extract(self, instance: Topic) -> NormalizedDocument:
            titles = [course.title for course in instance.courses.order_by("pk")]
            return self.build_document(
                instance, text=f"{instance.title}: {', '.join(titles)}"
            )


def _register_venues_following_their_seminars() -> None:
    """Register only Venue, by its name, following its seminars by the reverse of
    the multi-column ForeignObject ``venue``: Seminar itself is not."""
    rag.register(Venue, fields=["name"], follow=["seminars"])


def _register_venues_by_their_seminar_titles() -> None:
    """Register only Venue, with a custom extractor reading its name, then the
    titles of its seminars: depends_on names the reverse of the multi-column
    ForeignObject ``venue``, ``seminars``, whose saves change its documents.
    Seminar itself is not registered."""

    @rag.register_extractor(Venue, depends_on=["seminars"])
    class VenueExtractor(BaseExtractor[Venue]):
        def extract(self, instance: Venue) -> NormalizedDocument:
            titles = [seminar.title for seminar in instance.seminars.order_by("pk")]
            return self.build_document(
                instance, text="\n\n".join([instance.name, *titles])
            )


def _register_suppliers_following_their_profile() -> None:
    """Register only Supplier, following its profile by the reverse one-to-one
    accessor ``profile``: SupplierProfile itself is not."""
    rag.register(Supplier, follow=["profile"])


def _register_suppliers_by_their_profile_body() -> None:
    """Register only Supplier, with a custom extractor reading its name, then its
    profile's body if it has one: depends_on names the reverse one-to-one by its
    accessor, ``profile``, while the relation's query name is
    ``supplier_profile``. SupplierProfile itself is not registered."""

    @rag.register_extractor(Supplier, depends_on=["profile"])
    class SupplierExtractor(BaseExtractor[Supplier]):
        def extract(self, instance: Supplier) -> NormalizedDocument:
            try:
                body = instance.profile.body
            except SupplierProfile.DoesNotExist:
                return self.build_document(instance, text=instance.name)
            return self.build_document(instance, text=f"{instance.name}\n\n{body}")


def _register_warehouses_following_their_shelves() -> None:
    """Register only Warehouse, following its shelves, whose foreign key holds
    its code, not its primary key: Shelf itself is not."""
    rag.register(Warehouse, follow=["shelves"])


def _register_guilds_following_their_members() -> None:
    """Register only Guild, by its name, following its members through its own
    many-to-many ``members``: Craftsman itself is not."""
    rag.register(Guild, fields=["name"], follow=["members"])


def _register_musicians_following_their_bands() -> None:
    """Register only Musician, by its name, following its bands through the
    reverse many-to-many ``bands``: Band itself is not."""
    rag.register(Musician, fields=["name"], follow=["bands"])


def _register_persons_following_their_mentees() -> None:
    """Register Person, by its name, following its mentees, the reverse side of
    its own many-to-many ``mentors``: both sides of the links are Persons."""
    rag.register(Person, fields=["name"], follow=["mentees"])


def _register_persons_following_their_mentors() -> None:
    """Register Person, by its name, following its mentors, the forward side of
    its own many-to-many ``mentors``: both sides of the links are Persons."""
    rag.register(Person, fields=["name"], follow=["mentors"])


def _register_teams_following_their_matches() -> None:
    """Register Team, by its name, following its matches by both reverse foreign
    keys, ``home_matches`` and ``away_matches``: Match itself is not."""
    rag.register(Team, fields=["name"], follow=["home_matches", "away_matches"])


def _create_the_hall_and_its_acoustics_seminar() -> tuple[Venue, Seminar]:
    """Create the Halle Tony Garnier, a venue of Lyon, and its Acoustics seminar."""
    hall = _create_the_halle_tony_garnier()
    return hall, _create_a_seminar_at(hall)


def _create_the_transbordeur_and_its_seminar() -> Seminar:
    """Create the Transbordeur, a venue of Lyon, and its Stage lighting seminar."""
    transbordeur = _create_the_transbordeur()
    return _create_a_seminar_at(transbordeur, title="Stage lighting")


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


def _create_a_clearance_desk_lamp(category: Category) -> ClearanceProduct:
    """Create a Desk lamp, a ClearanceProduct: a Product row and its child row,
    whose primary key is its own code, ``CLR-1``, not the Product's."""
    return ClearanceProduct.objects.create(
        code="CLR-1",
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


def _create_the_carving_topic() -> Topic:
    """Create the Carving topic."""
    return Topic.objects.create(
        summary="Knives and gouges.", title="Carving", slug="carving"
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
        lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
    built_outputs.clear()

    # The child's primary key is its code, not the Product's: the parent row
    # is reached by the explicit parent link, ``product``.
    with django_capture_on_commit_callbacks(execute=True):
        lamp = _create_a_clearance_desk_lamp(lighting)
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
        lighting = _create_the_lighting_category()
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


# The model of the follower rows, and the model of the row holding the key.
FollowerT = TypeVar("FollowerT", bound=Model)
HolderT = TypeVar("HolderT", bound=Model)
# The model of the row whose groups are built: a follower, or a holding row.
RowT_contra = TypeVar("RowT_contra", bound=Model, contravariant=True)


class _GroupsOf(Protocol[RowT_contra]):
    """The groups a row leaves as committed, given the texts the holding rows on
    it were created with, in the order they were created: ``holding`` is
    keyword-only, so that each write says which rows are on it."""

    def __call__(
        self, row: RowT_contra, /, *, holding: Sequence[str]
    ) -> dict[str, list[NormalizedDocument]]: ...


def _no_other_groups(
    holder: Model, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """A holding row that names its follower alone: no other group."""
    return {}


@dataclass(frozen=True, kw_only=True)
class _FollowedCase(Generic[FollowerT, HolderT]):
    """What a declaration and a relation provide to the writes of a row holding
    the key to its follower: the writes themselves are generic over cases."""

    # Registers the follower's model and its declaration.
    register: Callable[[], None]
    # The follower the holding row is first written on.
    create_follower: Callable[[], FollowerT]
    # Another follower, the holding row is moved to.
    create_other_follower: Callable[[], FollowerT]
    # Creates the other rows the holding row names, if any, before anything
    # else, and returns the function creating a holding row on a follower,
    # told apart from the others by the given text.
    prepare_holder: Callable[[], Callable[[FollowerT, str], HolderT]]
    # The text the holding row the writes are about is created with.
    holder_text: str
    # The text of a sibling holding row, created by the same function on the
    # follower the holding row is first written on, before it, and never
    # written: None for a relation that allows one holding row per follower.
    sibling_text: str | None
    # Points the holding row's key to the given follower, without saving it.
    point_holder_to: Callable[[HolderT, FollowerT], None]
    # The follower's group as committed, under its source key, given the
    # holding rows its key points to.
    group_of: _GroupsOf[FollowerT]
    # The groups of the other rows the holding row names, whose keys no write
    # changes, as committed, given the holding rows still naming them, the
    # sibling included: none for a holding row that names its follower alone.
    groups_of_the_others_named: _GroupsOf[HolderT] = _no_other_groups


def _naming_the_follower_alone(
    create_holder: Callable[[FollowerT, str], HolderT],
) -> Callable[[], Callable[[FollowerT, str], HolderT]]:
    """For a holding row that names its follower alone: no other row to create
    first, a holding row created on a follower by the given function."""
    return lambda: create_holder


def _create_the_about_page() -> Page:
    """Create the About us page."""
    return Page.objects.create(title="About us", slug="about-us")


def _create_the_workshop_page() -> Page:
    """Create the Our workshop page."""
    return Page.objects.create(title="Our workshop", slug="our-workshop")


# The body a text plugin is created with unless another one is given, and the
# body of its sibling on the same page.
_PLUGIN_BODY = "We build chairs by hand."
_SIBLING_PLUGIN_BODY = "We ship worldwide."


def _create_a_plugin_on(page: Page, body: str = _PLUGIN_BODY) -> TextPlugin:
    """Create a text plugin with the given body, _PLUGIN_BODY by default, on the
    given page: it holds the key to the page."""
    return TextPlugin.objects.create(page=page, body=body)


def _point_the_plugin_to(plugin: TextPlugin, page: Page) -> None:
    """Point the plugin's foreign key to the given page, without saving it."""
    plugin.page = page


def _group_of_a_page_following_its_plugins(
    page: Page, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The page's group: its title, then the text of each plugin on the page."""
    return {
        f"testapp.page:{page.pk}": [
            NormalizedDocument(
                text="\n\n".join([page.title, *holding]),
                source_app_label="testapp",
                source_model="page",
                source_pk=page.pk,
                title=page.title,
                url=f"/pages/{page.slug}/",
            ),
        ],
    }


# `follow` through a reverse foreign key: only Page is registered, following
# its text plugins (TextPlugin is not); a plugin holds the key to its Page.
FOLLOW_REVERSE_FOREIGN_KEY: _FollowedCase[Page, TextPlugin] = _FollowedCase(
    register=_register_pages_following_their_plugins,
    create_follower=_create_the_about_page,
    create_other_follower=_create_the_workshop_page,
    prepare_holder=_naming_the_follower_alone(_create_a_plugin_on),
    holder_text=_PLUGIN_BODY,
    sibling_text=_SIBLING_PLUGIN_BODY,
    point_holder_to=_point_the_plugin_to,
    group_of=_group_of_a_page_following_its_plugins,
)


def _group_of_a_page_by_its_plugin_bodies(
    page: Page, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The page's group, as its custom extractor builds it: its title, then the
    body of each plugin on the page; no title or URL of its own."""
    return {
        f"testapp.page:{page.pk}": [
            NormalizedDocument(
                text="\n\n".join([page.title, *holding]),
                source_app_label="testapp",
                source_model="page",
                source_pk=page.pk,
            ),
        ],
    }


# `depends_on` through a reverse foreign key: only Page is registered, with a
# custom extractor reading its plugins' bodies (TextPlugin is not); a plugin
# holds the key to its Page.
DEPENDS_ON_REVERSE_FOREIGN_KEY: _FollowedCase[Page, TextPlugin] = _FollowedCase(
    register=_register_pages_by_their_plugin_bodies,
    create_follower=_create_the_about_page,
    create_other_follower=_create_the_workshop_page,
    prepare_holder=_naming_the_follower_alone(_create_a_plugin_on),
    holder_text=_PLUGIN_BODY,
    sibling_text=_SIBLING_PLUGIN_BODY,
    point_holder_to=_point_the_plugin_to,
    group_of=_group_of_a_page_by_its_plugin_bodies,
)


def _create_birch_mill() -> Supplier:
    """Create the Birch Mill supplier."""
    return Supplier.objects.create(name="Birch Mill")


def _create_oak_yard() -> Supplier:
    """Create the Oak Yard supplier."""
    return Supplier.objects.create(name="Oak Yard")


# The body a supplier's profile is created with.
_PROFILE_BODY = "Kiln-dried boards."


def _create_a_profile_of(supplier: Supplier, body: str) -> SupplierProfile:
    """Create the given supplier's profile, with the given body: it holds the key
    to the supplier."""
    return SupplierProfile.objects.create(supplier=supplier, body=body)


def _point_the_profile_to(profile: SupplierProfile, supplier: Supplier) -> None:
    """Point the profile's one-to-one key to the given supplier, without saving
    it."""
    profile.supplier = supplier


def _group_of_a_supplier_following_its_profile(
    supplier: Supplier, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The supplier's group: its name, then the profile's body if the profile
    is on the supplier."""
    return {
        f"testapp.supplier:{supplier.pk}": [
            NormalizedDocument(
                text="\n\n".join([supplier.name, *holding]),
                source_app_label="testapp",
                source_model="supplier",
                source_pk=supplier.pk,
                title=supplier.name,
            ),
        ],
    }


# `follow` through a reverse one-to-one: only Supplier is registered, following
# its profile (SupplierProfile is not); a profile holds the key to its Supplier.
FOLLOW_REVERSE_ONE_TO_ONE: _FollowedCase[Supplier, SupplierProfile] = _FollowedCase(
    register=_register_suppliers_following_their_profile,
    create_follower=_create_birch_mill,
    create_other_follower=_create_oak_yard,
    prepare_holder=_naming_the_follower_alone(_create_a_profile_of),
    holder_text=_PROFILE_BODY,
    # A supplier has at most one profile.
    sibling_text=None,
    point_holder_to=_point_the_profile_to,
    group_of=_group_of_a_supplier_following_its_profile,
)


def _group_of_a_supplier_by_its_profile_body(
    supplier: Supplier, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The supplier's group, as its custom extractor builds it: its name, then
    the profile's body if the profile is on the supplier; no title of its own."""
    return {
        f"testapp.supplier:{supplier.pk}": [
            NormalizedDocument(
                text="\n\n".join([supplier.name, *holding]),
                source_app_label="testapp",
                source_model="supplier",
                source_pk=supplier.pk,
            ),
        ],
    }


# `depends_on` through a reverse one-to-one: only Supplier is registered, with a
# custom extractor reading its profile's body by the accessor `profile`
# (SupplierProfile is not); a profile holds the key to its Supplier.
DEPENDS_ON_REVERSE_ONE_TO_ONE: _FollowedCase[Supplier, SupplierProfile] = _FollowedCase(
    register=_register_suppliers_by_their_profile_body,
    create_follower=_create_birch_mill,
    create_other_follower=_create_oak_yard,
    prepare_holder=_naming_the_follower_alone(_create_a_profile_of),
    holder_text=_PROFILE_BODY,
    # A supplier has at most one profile.
    sibling_text=None,
    point_holder_to=_point_the_profile_to,
    group_of=_group_of_a_supplier_by_its_profile_body,
)


def _create_the_north_warehouse() -> Warehouse:
    """Create the North depot warehouse, coded "north"."""
    return Warehouse.objects.create(name="North depot", code="north")


def _create_the_south_warehouse() -> Warehouse:
    """Create the South depot warehouse, coded "south"."""
    return Warehouse.objects.create(name="South depot", code="south")


# The label a shelf is created with unless another one is given, and the label
# of its sibling in the same warehouse.
_SHELF_LABEL = "Timber"
_SIBLING_SHELF_LABEL = "Paint"


def _create_a_shelf_in(warehouse: Warehouse, label: str = _SHELF_LABEL) -> Shelf:
    """Create a shelf with the given label, _SHELF_LABEL by default, in the given
    warehouse: its foreign key holds the warehouse's code, not its primary
    key."""
    return Shelf.objects.create(warehouse=warehouse, label=label)


def _point_the_shelf_to(shelf: Shelf, warehouse: Warehouse) -> None:
    """Point the shelf's foreign key to the given warehouse's code, without
    saving it."""
    shelf.warehouse = warehouse


def _group_of_a_warehouse_following_its_shelves(
    warehouse: Warehouse, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The warehouse's group, under its primary key, not its code: its name,
    then the label of each shelf in the warehouse."""
    return {
        f"testapp.warehouse:{warehouse.pk}": [
            NormalizedDocument(
                text="\n\n".join([warehouse.name, *holding]),
                source_app_label="testapp",
                source_model="warehouse",
                source_pk=warehouse.pk,
                title=warehouse.name,
            ),
        ],
    }


# `follow` through a reverse foreign key to a unique column: only Warehouse is
# registered, following its shelves (Shelf is not); a shelf holds the key to
# its Warehouse, the Warehouse's code rather than its primary key.
FOLLOW_REVERSE_FOREIGN_KEY_TO_FIELD: _FollowedCase[Warehouse, Shelf] = _FollowedCase(
    register=_register_warehouses_following_their_shelves,
    create_follower=_create_the_north_warehouse,
    create_other_follower=_create_the_south_warehouse,
    prepare_holder=_naming_the_follower_alone(_create_a_shelf_in),
    holder_text=_SHELF_LABEL,
    sibling_text=_SIBLING_SHELF_LABEL,
    point_holder_to=_point_the_shelf_to,
    group_of=_group_of_a_warehouse_following_its_shelves,
)


def _create_the_halle_tony_garnier() -> Venue:
    """Create the Halle Tony Garnier, a venue of Lyon."""
    return Venue.objects.create(city="Lyon", name="Halle Tony Garnier")


def _create_the_transbordeur() -> Venue:
    """Create the Transbordeur, another venue of Lyon: only the name column
    tells it apart from the Halle Tony Garnier."""
    return Venue.objects.create(city="Lyon", name="Transbordeur")


# The title a seminar is created with unless another one is given, and the
# title of its sibling at the same venue.
_SEMINAR_TITLE = "Acoustics"
_SIBLING_SEMINAR_TITLE = "Stage lighting"


def _create_a_seminar_at(venue: Venue, title: str = _SEMINAR_TITLE) -> Seminar:
    """Create a seminar with the given title, _SEMINAR_TITLE by default, at the
    given venue: its two columns hold the venue's city and name."""
    return Seminar.objects.create(
        title=title, venue_city=venue.city, venue_name=venue.name
    )


def _point_the_seminar_to(seminar: Seminar, venue: Venue) -> None:
    """Point the seminar's two columns to the given venue's city and name,
    without saving it."""
    seminar.venue_city = venue.city
    seminar.venue_name = venue.name


def _group_of_a_venue_following_its_seminars(
    venue: Venue, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The venue's group: its name, then the text fields of each seminar at the
    venue, title first."""
    seminars = [f"{title}\n\n{venue.city}\n\n{venue.name}" for title in holding]
    return {
        f"testapp.venue:{venue.pk}": [
            NormalizedDocument(
                text="\n\n".join([venue.name, *seminars]),
                source_app_label="testapp",
                source_model="venue",
                source_pk=venue.pk,
                title=venue.name,
            ),
        ],
    }


# `follow` through the reverse of a multi-column relation: only Venue is
# registered, following its seminars (Seminar is not); a seminar holds the key
# to its Venue in two columns, matched to the Venue's city and name.
FOLLOW_REVERSE_MULTI_COLUMN: _FollowedCase[Venue, Seminar] = _FollowedCase(
    register=_register_venues_following_their_seminars,
    create_follower=_create_the_halle_tony_garnier,
    create_other_follower=_create_the_transbordeur,
    prepare_holder=_naming_the_follower_alone(_create_a_seminar_at),
    holder_text=_SEMINAR_TITLE,
    sibling_text=_SIBLING_SEMINAR_TITLE,
    point_holder_to=_point_the_seminar_to,
    group_of=_group_of_a_venue_following_its_seminars,
)


def _group_of_a_venue_by_its_seminar_titles(
    venue: Venue, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The venue's group, as its custom extractor builds it: its name, then the
    title of each seminar at the venue; no title of its own."""
    return {
        f"testapp.venue:{venue.pk}": [
            NormalizedDocument(
                text="\n\n".join([venue.name, *holding]),
                source_app_label="testapp",
                source_model="venue",
                source_pk=venue.pk,
            ),
        ],
    }


# `depends_on` through the reverse of a multi-column relation: only Venue is
# registered, with a custom extractor reading its seminars' titles (Seminar is
# not); a seminar holds the key to its Venue in two columns, matched to the
# Venue's city and name.
DEPENDS_ON_REVERSE_MULTI_COLUMN: _FollowedCase[Venue, Seminar] = _FollowedCase(
    register=_register_venues_by_their_seminar_titles,
    create_follower=_create_the_halle_tony_garnier,
    create_other_follower=_create_the_transbordeur,
    prepare_holder=_naming_the_follower_alone(_create_a_seminar_at),
    holder_text=_SEMINAR_TITLE,
    sibling_text=_SIBLING_SEMINAR_TITLE,
    point_holder_to=_point_the_seminar_to,
    group_of=_group_of_a_venue_by_its_seminar_titles,
)


def _create_lyon() -> Team:
    """Create the Lyon team."""
    return Team.objects.create(name="Lyon")


def _create_marseille() -> Team:
    """Create the Marseille team."""
    return Team.objects.create(name="Marseille")


def _create_nantes() -> Team:
    """Create the Nantes team, the opponent of every match."""
    return Team.objects.create(name="Nantes")


# The title a match is created with unless another one is given, and the title
# of its sibling, the same teams on the same sides.
_MATCH_TITLE = "Opening day"
_SIBLING_MATCH_TITLE = "Rematch"


def _create_a_match(
    *,
    home_team: Team,
    away_team: Team,
    tournament: Tournament | None = None,
    title: str = _MATCH_TITLE,
) -> Match:
    """Create a match with the given title, _MATCH_TITLE by default, between the
    given teams, in the given tournament if any: it holds the key to two
    different teams, one per foreign key."""
    return Match.objects.create(
        title=title,
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
    )


def _group_of_a_team_following_its_matches(
    team: Team, *, holding: Sequence[str]
) -> dict[str, list[NormalizedDocument]]:
    """The team's group: its name, then the title of each match naming the team
    on one side, its matches at home before those away."""
    return {
        f"testapp.team:{team.pk}": [
            NormalizedDocument(
                text="\n\n".join([team.name, *holding]),
                source_app_label="testapp",
                source_model="team",
                source_pk=team.pk,
                title=team.name,
            ),
        ],
    }


class _Side(Enum):
    """The side a team plays a match on, by the name of the match's foreign key
    to it."""

    HOME = "home_team"
    AWAY = "away_team"

    @property
    def opposite(self) -> "_Side":
        """The side its opponent plays on."""
        return _Side.AWAY if self is _Side.HOME else _Side.HOME


def _teams_following_their_matches_on(side: _Side) -> _FollowedCase[Team, Match]:
    """The case of a team following its matches, whose holding row is a match
    with the follower on the given side, against Nantes on the opposite one:
    its sibling too, so Nantes's group holds the sibling's title as well, and
    both matches are on the same relation of each team."""

    def create_nantes_to_play_against() -> Callable[[Team, str], Match]:
        nantes = _create_nantes()

        def create_a_match_of(team: Team, title: str) -> Match:
            home_team, away_team = (
                (team, nantes) if side is _Side.HOME else (nantes, team)
            )
            return _create_a_match(
                home_team=home_team, away_team=away_team, title=title
            )

        return create_a_match_of

    def point_the_match_to(match: Match, team: Team) -> None:
        # Only the follower's key: Nantes's stays.
        setattr(match, side.value, team)

    def group_of_nantes(
        match: Match, *, holding: Sequence[str]
    ) -> dict[str, list[NormalizedDocument]]:
        nantes: Team = getattr(match, side.opposite.value)
        return _group_of_a_team_following_its_matches(nantes, holding=holding)

    return _FollowedCase(
        register=_register_teams_following_their_matches,
        create_follower=_create_lyon,
        create_other_follower=_create_marseille,
        prepare_holder=create_nantes_to_play_against,
        holder_text=_MATCH_TITLE,
        sibling_text=_SIBLING_MATCH_TITLE,
        point_holder_to=point_the_match_to,
        group_of=_group_of_a_team_following_its_matches,
        groups_of_the_others_named=group_of_nantes,
    )


# `follow` through two reverse foreign keys to the same model: only Team is
# registered, following its matches at home and away (Match is not); a match
# holds the key to two different teams, the follower at home and Nantes away,
# created first and handed to every match, so each relation must be crossed
# for its team's group to be replaced.
FOLLOW_TWO_REVERSE_FOREIGN_KEYS: _FollowedCase[Team, Match] = (
    _teams_following_their_matches_on(_Side.HOME)
)


# The mirror of the case above: the follower plays away and Nantes at home,
# so the write changes the key of the reverse relation ``away_matches``.
FOLLOW_TWO_REVERSE_FOREIGN_KEYS_AWAY: _FollowedCase[Team, Match] = (
    _teams_following_their_matches_on(_Side.AWAY)
)


# A write performed in the captured callbacks, returning the replace calls it
# must send once its transaction commits.
Act = Callable[[], ReplaceCalls]


class _Write(Protocol):
    """A write of a holding row, generic over the case it is performed on."""

    def __call__(self, case: _FollowedCase[FollowerT, HolderT], /) -> Act: ...


def _merged(
    *groups: dict[str, list[NormalizedDocument]],
) -> dict[str, list[NormalizedDocument]]:
    """The given groups in one mapping, once checked that no two share a source
    key: merging them blindly would drop one of the colliding groups, shrinking
    the expected call to match a wrong one."""
    merged: dict[str, list[NormalizedDocument]] = {}
    for group in groups:
        shared = merged.keys() & group.keys()
        assert not shared, f"groups expected under the same key: {sorted(shared)}"
        merged |= group
    return merged


def _create_the_sibling_on(
    follower: FollowerT,
    case: _FollowedCase[FollowerT, HolderT],
    create_holder: Callable[[FollowerT, str], HolderT],
) -> list[str]:
    """Create the case's sibling holding row on the given follower, if its
    relation allows one, and return the texts of the holding rows it leaves
    there: the sibling's, or none."""
    if case.sibling_text is None:
        return []
    create_holder(follower, case.sibling_text)
    return [case.sibling_text]


def _create(case: _FollowedCase[FollowerT, HolderT]) -> Act:
    """The holding row created on a follower: one call, the follower's group and
    those of the other rows it names, each once."""
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the holding row's creation is.
    create_holder = case.prepare_holder()
    follower = case.create_follower()
    siblings = _create_the_sibling_on(follower, case, create_holder)

    def act() -> ReplaceCalls:
        holder = create_holder(follower, case.holder_text)
        # The follower and the other rows it names gaining the holding row,
        # after the sibling, created first.
        holding = [*siblings, case.holder_text]
        return [
            _merged(
                case.group_of(follower, holding=holding),
                case.groups_of_the_others_named(holder, holding=holding),
            )
        ]

    return act


def _move(case: _FollowedCase[FollowerT, HolderT]) -> Act:
    """The holding row moved to another follower by ``save()``: one call, both
    followers' groups and those of the other rows it names, each once."""
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the move is.
    create_holder = case.prepare_holder()
    old = case.create_follower()
    new = case.create_other_follower()
    siblings = _create_the_sibling_on(old, case, create_holder)
    holder = create_holder(old, case.holder_text)

    def act() -> ReplaceCalls:
        case.point_holder_to(holder, new)
        holder.save()
        # The old follower left with the sibling alone, the new one gaining the
        # holding row, the other rows it names keeping both.
        return [
            _merged(
                case.group_of(old, holding=siblings),
                case.group_of(new, holding=[case.holder_text]),
                case.groups_of_the_others_named(
                    holder, holding=[*siblings, case.holder_text]
                ),
            )
        ]

    return act


def _delete(case: _FollowedCase[FollowerT, HolderT]) -> Act:
    """The holding row deleted: one call, the follower's group and those of the
    other rows it named, each once."""
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the delete is.
    create_holder = case.prepare_holder()
    follower = case.create_follower()
    siblings = _create_the_sibling_on(follower, case, create_holder)
    holder = create_holder(follower, case.holder_text)

    def act() -> ReplaceCalls:
        holder.delete()
        # The follower and the other rows it named left with the sibling alone.
        return [
            _merged(
                case.group_of(follower, holding=siblings),
                case.groups_of_the_others_named(holder, holding=siblings),
            )
        ]

    return act


@pytest.mark.django_db
@pytest.mark.parametrize(
    "write",
    [
        pytest.param(_create, id="create"),
        pytest.param(_move, id="move"),
        pytest.param(_delete, id="delete"),
    ],
)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(FOLLOW_REVERSE_FOREIGN_KEY, id="follow-reverse_foreign_key"),
        pytest.param(
            DEPENDS_ON_REVERSE_FOREIGN_KEY, id="depends_on-reverse_foreign_key"
        ),
        pytest.param(FOLLOW_REVERSE_ONE_TO_ONE, id="follow-reverse_one_to_one"),
        pytest.param(DEPENDS_ON_REVERSE_ONE_TO_ONE, id="depends_on-reverse_one_to_one"),
        pytest.param(
            FOLLOW_REVERSE_FOREIGN_KEY_TO_FIELD,
            id="follow-reverse_foreign_key_to_field",
        ),
        pytest.param(FOLLOW_REVERSE_MULTI_COLUMN, id="follow-reverse_multi_column"),
        pytest.param(
            DEPENDS_ON_REVERSE_MULTI_COLUMN, id="depends_on-reverse_multi_column"
        ),
        pytest.param(
            FOLLOW_TWO_REVERSE_FOREIGN_KEYS, id="follow-two_reverse_foreign_keys"
        ),
        pytest.param(
            FOLLOW_TWO_REVERSE_FOREIGN_KEYS_AWAY,
            id="follow-two_reverse_foreign_keys_away",
        ),
    ],
)
def test_writing_a_row_holding_the_key_to_its_follower_replaces_the_followers_group(
    # Any: the cases pair different models, and _FollowedCase is invariant in
    # both, so no single precise type covers them all.
    case: _FollowedCase[Any, Any],
    write: _Write,
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The case registers its models; the write creates the rows it needs,
    # outside the captured callbacks.
    case.register()
    act = write(case)

    with django_capture_on_commit_callbacks(execute=True):
        expected = act()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Exactly the replace calls the write must send, in call order.
    assert _replaced(built_outputs) == expected


# The model of the row a follower's path ends on, the row written.
TargetT = TypeVar("TargetT", bound=Model)
# The model of the follower whose group is built, and of the target it is on.
FollowerT_contra = TypeVar("FollowerT_contra", bound=Model, contravariant=True)
TargetT_contra = TypeVar("TargetT_contra", bound=Model, contravariant=True)


class _GroupOnTarget(Protocol[FollowerT_contra, TargetT_contra]):
    """The group a follower leaves as committed, its path ending on the given
    target: ``target`` is keyword-only, so that each write says which."""

    def __call__(
        self, follower: FollowerT_contra, /, *, target: TargetT_contra
    ) -> dict[str, list[NormalizedDocument]]: ...


@dataclass(frozen=True, kw_only=True)
class _KeyChange(Generic[TargetT, FollowerT]):
    """A change of the column a follower points to its target by, and the
    follower's group it leaves as committed."""

    # Changes that column on the target and saves it; then, for a relation
    # whose followers move with it, moves them to the new value by a queryset
    # update, which sends no signal.
    change: Callable[[TargetT], None]
    # The follower's group as committed, under its source key, once its
    # target's key changed: on the target, for a follower moved with it; its
    # own fields alone, for a follower left naming the old key.
    group_of: _GroupOnTarget[FollowerT, TargetT]


@dataclass(frozen=True, kw_only=True)
class _TargetCase(Generic[TargetT, FollowerT]):
    """What a declaration and a relation provide to the writes of a row its
    follower's path ends on: the writes themselves are generic over cases."""

    # Registers the follower's model and its declaration.
    register: Callable[[], None]
    # The target the writes are performed on.
    create_target: Callable[[], TargetT]
    # Another target, with a follower of its own, that no write touches: it is
    # created knowing the target, for a case whose other target must name it.
    create_other_target: Callable[[TargetT], TargetT]
    # Creates a follower whose path ends on the given target.
    create_follower_on: Callable[[TargetT], FollowerT]
    # Changes a value of the target its follower reads, without saving it.
    change_target: Callable[[TargetT], None]
    # The follower's group as committed, under its source key, its path ending
    # on the given target.
    group_of: _GroupOnTarget[FollowerT, TargetT]
    # The follower's group as committed once its target is deleted, under its
    # source key: what is left depends on the relation's on_delete.
    group_once_target_deleted: Callable[
        [FollowerT], dict[str, list[NormalizedDocument]]
    ]
    # For a follower pointing to its target by a column other than its primary
    # key only: the change of that column, and the follower's group it leaves.
    # A case that provides it gets the "change key" write too.
    change_key: _KeyChange[TargetT, FollowerT] | None = None


def _whatever_the_target(
    create_other_target: Callable[[], TargetT],
) -> Callable[[TargetT], TargetT]:
    """For another target that does not depend on the target: created by the
    given function, the target ignored."""
    return lambda _target: create_other_target()


def _create_the_lighting_category() -> Category:
    """Create the Lighting category."""
    return Category.objects.create(name="Lighting")


def _create_the_tools_category() -> Category:
    """Create the Tools category."""
    return Category.objects.create(name="Tools")


def _rename_the_category(category: Category) -> None:
    """Rename the given category to Lamps, without saving it."""
    category.name = "Lamps"


def _register_products_following_their_category() -> None:
    """Register only Product, by its name, following its category through its
    own foreign key: Category itself is not registered."""
    rag.register(Product, fields=["name"], follow=["category"])


def _group_of_a_desk_lamp_reading_its_category(
    product: Product, /, *, target: Category
) -> dict[str, list[NormalizedDocument]]:
    """The desk lamp's group: its name, then its category's name, whether it
    follows its category or reads its name through a lookup path."""
    return {
        f"testapp.product:{product.pk}": [
            NormalizedDocument(
                text=f"Desk lamp\n\n{target.name}",
                source_app_label="testapp",
                source_model="product",
                source_pk=product.pk,
                title="Desk lamp",
                url=f"/products/{product.pk}/",
            ),
        ],
    }


def _group_of_a_product_deleted_by_cascade(
    product: Product,
) -> dict[str, list[NormalizedDocument]]:
    """The product's group once deleted with its category by cascade: empty."""
    return {f"testapp.product:{product.pk}": []}


# `follow` through a forward foreign key: only Product is registered,
# following its category (Category is not); the product holds the key to the
# category, the row written, CASCADE on delete.
FOLLOW_FORWARD_FOREIGN_KEY: _TargetCase[Category, Product] = _TargetCase(
    register=_register_products_following_their_category,
    create_target=_create_the_lighting_category,
    create_other_target=_whatever_the_target(_create_the_tools_category),
    create_follower_on=_create_a_plain_desk_lamp,
    change_target=_rename_the_category,
    group_of=_group_of_a_desk_lamp_reading_its_category,
    group_once_target_deleted=_group_of_a_product_deleted_by_cascade,
)


def _retitle_the_topic(topic: Topic) -> None:
    """Retitle the given topic to Joinery, without saving it."""
    topic.title = "Joinery"


def _register_workshops_following_their_topic() -> None:
    """Register Workshop, following its topic through its own nullable foreign
    key: Topic itself is not registered."""
    rag.register(Workshop, follow=["topic"])


def _create_a_pottery_workshop(topic: Topic) -> Workshop:
    """Create a Pottery workshop on the given topic."""
    return Workshop.objects.create(title="Pottery", topic=topic)


def _group_of_a_pottery_workshop_following_its_topic(
    workshop: Workshop, /, *, target: Topic
) -> dict[str, list[NormalizedDocument]]:
    """The pottery workshop's group: its title, then its topic's title and
    summary."""
    return {
        f"testapp.workshop:{workshop.pk}": [
            NormalizedDocument(
                text=f"Pottery\n\n{target.title}\n\n{target.summary}",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=workshop.pk,
                title="Pottery",
            ),
        ],
    }


def _group_of_a_pottery_workshop_whose_topic_is_nulled(
    workshop: Workshop,
) -> dict[str, list[NormalizedDocument]]:
    """The pottery workshop's group once its topic is deleted and its foreign
    key set to null: the topic's text is gone, only its own title is left."""
    return {
        f"testapp.workshop:{workshop.pk}": [
            NormalizedDocument(
                text="Pottery",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=workshop.pk,
                title="Pottery",
            ),
        ],
    }


# `follow` through a nullable forward foreign key: only Workshop is
# registered, following its topic (Topic is not); the workshop holds the key
# to the topic, the row written, SET_NULL on delete.
FOLLOW_FORWARD_FOREIGN_KEY_SET_NULL: _TargetCase[Topic, Workshop] = _TargetCase(
    register=_register_workshops_following_their_topic,
    create_target=_create_the_woodworking_topic,
    create_other_target=_whatever_the_target(_create_the_carving_topic),
    create_follower_on=_create_a_pottery_workshop,
    change_target=_retitle_the_topic,
    group_of=_group_of_a_pottery_workshop_following_its_topic,
    group_once_target_deleted=_group_of_a_pottery_workshop_whose_topic_is_nulled,
)


def _register_products_reading_their_category_through_a_path() -> None:
    """Register Product, reading its category's name through a lookup path,
    with no follow: Category itself is not registered."""
    rag.register(Product, fields=["name", "category__name"])


# A lookup path through a forward foreign key: only Product is registered,
# reading its category's name (Category is not); the product holds the key to
# the category, the row written, CASCADE on delete. The path alone builds the
# same group as follow=["category"].
PATH_FORWARD_FOREIGN_KEY: _TargetCase[Category, Product] = _TargetCase(
    register=_register_products_reading_their_category_through_a_path,
    create_target=_create_the_lighting_category,
    create_other_target=_whatever_the_target(_create_the_tools_category),
    create_follower_on=_create_a_plain_desk_lamp,
    change_target=_rename_the_category,
    group_of=_group_of_a_desk_lamp_reading_its_category,
    group_once_target_deleted=_group_of_a_product_deleted_by_cascade,
)


def _register_excerpts_reading_their_language_through_a_path() -> None:
    """Register Excerpt, reading its language from its notice through a lookup
    path, with no follow: Notice itself is not registered."""
    rag.register(Excerpt, fields=["title"], language_field="notice__language")


def _create_the_french_notice() -> Notice:
    """Create a notice in French."""
    return Notice.objects.create(title="Avis", language="fr")


def _create_the_english_notice() -> Notice:
    """Create a notice in English."""
    return Notice.objects.create(title="Notice", language="en")


def _create_a_greeting_excerpt(notice: Notice) -> Excerpt:
    """Create a Bonjour excerpt on the given notice."""
    return Excerpt.objects.create(title="Bonjour", notice=notice)


def _switch_the_notice_to_english(notice: Notice) -> None:
    """Switch the given notice's language to English, without saving it."""
    notice.language = "en"


def _group_of_a_greeting_excerpt_reading_its_notice(
    excerpt: Excerpt, /, *, target: Notice
) -> dict[str, list[NormalizedDocument]]:
    """The greeting excerpt's group: its title, in its notice's language."""
    return {
        f"testapp.excerpt:{excerpt.pk}": [
            NormalizedDocument(
                text="Bonjour",
                source_app_label="testapp",
                source_model="excerpt",
                source_pk=excerpt.pk,
                title="Bonjour",
                language=target.language,
            ),
        ],
    }


def _group_of_a_greeting_excerpt_whose_notice_is_nulled(
    excerpt: Excerpt,
) -> dict[str, list[NormalizedDocument]]:
    """The greeting excerpt's group once its notice is deleted and its foreign
    key set to null: with no notice left, it has no language."""
    return {
        f"testapp.excerpt:{excerpt.pk}": [
            NormalizedDocument(
                text="Bonjour",
                source_app_label="testapp",
                source_model="excerpt",
                source_pk=excerpt.pk,
                title="Bonjour",
                language=None,
            ),
        ],
    }


# A lookup path through a nullable forward foreign key: only Excerpt is
# registered, reading its language from its notice (Notice is not); the
# excerpt holds the key to the notice, the row written, SET_NULL on delete.
PATH_FORWARD_FOREIGN_KEY_SET_NULL: _TargetCase[Notice, Excerpt] = _TargetCase(
    register=_register_excerpts_reading_their_language_through_a_path,
    create_target=_create_the_french_notice,
    create_other_target=_whatever_the_target(_create_the_english_notice),
    create_follower_on=_create_a_greeting_excerpt,
    change_target=_switch_the_notice_to_english,
    group_of=_group_of_a_greeting_excerpt_reading_its_notice,
    group_once_target_deleted=_group_of_a_greeting_excerpt_whose_notice_is_nulled,
)


def _register_citations_reading_their_url_through_a_path() -> None:
    """Register Citation, reading its url from its bookmark's link through a
    lookup path, with no follow: Bookmark itself is not registered."""
    rag.register(Citation, fields=["title"], url_field="bookmark__link")


def _create_the_docs_bookmark() -> Bookmark:
    """Create the Docs bookmark, linking to /docs/a/."""
    return Bookmark.objects.create(title="Docs", link="/docs/a/")


def _create_the_example_bookmark() -> Bookmark:
    """Create the Example bookmark, linking to /docs/b/."""
    return Bookmark.objects.create(title="Example", link="/docs/b/")


def _create_a_linked_citation(bookmark: Bookmark) -> Citation:
    """Create a Linked citation of the given bookmark."""
    return Citation.objects.create(title="Linked", bookmark=bookmark)


def _relink_the_bookmark(bookmark: Bookmark) -> None:
    """Point the given bookmark's link to /docs/c/, without saving it."""
    bookmark.link = "/docs/c/"


def _group_of_a_linked_citation_reading_its_bookmark(
    citation: Citation, /, *, target: Bookmark
) -> dict[str, list[NormalizedDocument]]:
    """The linked citation's group: its title, with its bookmark's link as its
    url."""
    return {
        f"testapp.citation:{citation.pk}": [
            NormalizedDocument(
                text="Linked",
                source_app_label="testapp",
                source_model="citation",
                source_pk=citation.pk,
                title="Linked",
                url=target.link,
            ),
        ],
    }


def _group_of_a_linked_citation_whose_bookmark_is_nulled(
    citation: Citation,
) -> dict[str, list[NormalizedDocument]]:
    """The linked citation's group once its bookmark is deleted and its foreign
    key set to null: with no bookmark left, it has no url, an empty string."""
    return {
        f"testapp.citation:{citation.pk}": [
            NormalizedDocument(
                text="Linked",
                source_app_label="testapp",
                source_model="citation",
                source_pk=citation.pk,
                title="Linked",
                url="",
            ),
        ],
    }


# A lookup path read as the url, through a nullable forward foreign key: only
# Citation is registered, reading its url from its bookmark's link (Bookmark is
# not); the citation holds the key to the bookmark, the row written, SET_NULL
# on delete.
PATH_FORWARD_FOREIGN_KEY_SET_NULL_AS_URL: _TargetCase[Bookmark, Citation] = _TargetCase(
    register=_register_citations_reading_their_url_through_a_path,
    create_target=_create_the_docs_bookmark,
    create_other_target=_whatever_the_target(_create_the_example_bookmark),
    create_follower_on=_create_a_linked_citation,
    change_target=_relink_the_bookmark,
    group_of=_group_of_a_linked_citation_reading_its_bookmark,
    group_once_target_deleted=_group_of_a_linked_citation_whose_bookmark_is_nulled,
)


def _register_citations_reading_their_title_through_a_path() -> None:
    """Register Citation, by its title, reading its document title from its
    bookmark's title through a lookup path, with no follow: Bookmark itself is
    not registered."""
    rag.register(Citation, fields=["title"], title_field="bookmark__title")


def _retitle_the_bookmark(bookmark: Bookmark) -> None:
    """Retitle the given bookmark to Guides, without saving it."""
    bookmark.title = "Guides"


def _group_of_a_linked_citation_titled_by_its_bookmark(
    citation: Citation, /, *, target: Bookmark
) -> dict[str, list[NormalizedDocument]]:
    """The linked citation's group: its own title as its text, with its
    bookmark's title as its document title."""
    return {
        f"testapp.citation:{citation.pk}": [
            NormalizedDocument(
                text="Linked",
                source_app_label="testapp",
                source_model="citation",
                source_pk=citation.pk,
                title=target.title,
            ),
        ],
    }


def _group_of_a_linked_citation_whose_title_bookmark_is_nulled(
    citation: Citation,
) -> dict[str, list[NormalizedDocument]]:
    """The linked citation's group once its bookmark is deleted and its foreign
    key set to null: with no bookmark left, it has no title, an empty
    string."""
    return {
        f"testapp.citation:{citation.pk}": [
            NormalizedDocument(
                text="Linked",
                source_app_label="testapp",
                source_model="citation",
                source_pk=citation.pk,
                title="",
            ),
        ],
    }


# A lookup path read as the title, through a nullable forward foreign key: only
# Citation is registered, reading its title from its bookmark's title (Bookmark
# is not); the citation holds the key to the bookmark, the row written,
# SET_NULL on delete.
PATH_FORWARD_FOREIGN_KEY_SET_NULL_AS_TITLE: _TargetCase[Bookmark, Citation] = (
    _TargetCase(
        register=_register_citations_reading_their_title_through_a_path,
        create_target=_create_the_docs_bookmark,
        create_other_target=_whatever_the_target(_create_the_example_bookmark),
        create_follower_on=_create_a_linked_citation,
        change_target=_retitle_the_bookmark,
        group_of=_group_of_a_linked_citation_titled_by_its_bookmark,
        group_once_target_deleted=(
            _group_of_a_linked_citation_whose_title_bookmark_is_nulled
        ),
    )
)


def _retitle_the_page(page: Page) -> None:
    """Retitle the given page to Our story, without saving it."""
    page.title = "Our story"


def _group_of_a_plugin_by_its_page_title(
    plugin: TextPlugin, /, *, target: Page
) -> dict[str, list[NormalizedDocument]]:
    """The plugin's group, as its custom extractor builds it: its page's title,
    then its body; no title or URL of its own."""
    return {
        f"testapp.textplugin:{plugin.pk}": [
            NormalizedDocument(
                text=f"{target.title}: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=plugin.pk,
            ),
        ],
    }


def _group_of_a_plugin_deleted_by_cascade(
    plugin: TextPlugin,
) -> dict[str, list[NormalizedDocument]]:
    """The plugin's group once deleted with its page by cascade: empty."""
    return {f"testapp.textplugin:{plugin.pk}": []}


# `depends_on` through a forward foreign key: only TextPlugin is registered,
# with a custom extractor reading its page's title (Page is not); the plugin
# holds the key to the page, the row written, CASCADE on delete.
DEPENDS_ON_FORWARD_FOREIGN_KEY: _TargetCase[Page, TextPlugin] = _TargetCase(
    register=_register_plugins_by_their_page_title,
    create_target=_create_the_about_page,
    create_other_target=_whatever_the_target(_create_the_workshop_page),
    create_follower_on=_create_a_plugin_on,
    change_target=_retitle_the_page,
    group_of=_group_of_a_plugin_by_its_page_title,
    group_once_target_deleted=_group_of_a_plugin_deleted_by_cascade,
)


def _register_workshops_by_their_topic_title() -> None:
    """Register only Workshop, with a custom extractor reading its topic's
    title before its own when it has a topic: depends_on names the topic,
    its own nullable foreign key. Topic itself is not registered."""

    @rag.register_extractor(Workshop, depends_on=["topic"])
    class WorkshopExtractor(BaseExtractor[Workshop]):
        def extract(self, instance: Workshop) -> NormalizedDocument:
            if instance.topic is None:
                return self.build_document(instance, text=instance.title)
            return self.build_document(
                instance, text=f"{instance.topic.title}: {instance.title}"
            )


def _group_of_a_pottery_workshop_by_its_topic_title(
    workshop: Workshop, /, *, target: Topic
) -> dict[str, list[NormalizedDocument]]:
    """The pottery workshop's group, as its custom extractor builds it: its
    topic's title, then its own; no title of its own in the document."""
    return {
        f"testapp.workshop:{workshop.pk}": [
            NormalizedDocument(
                text=f"{target.title}: Pottery",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=workshop.pk,
            ),
        ],
    }


def _group_of_a_pottery_workshop_by_its_nulled_topic(
    workshop: Workshop,
) -> dict[str, list[NormalizedDocument]]:
    """The pottery workshop's group, as its custom extractor builds it once its
    topic is deleted and its foreign key set to null: its own title alone."""
    return {
        f"testapp.workshop:{workshop.pk}": [
            NormalizedDocument(
                text="Pottery",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=workshop.pk,
            ),
        ],
    }


# `depends_on` through a nullable forward foreign key: only Workshop is
# registered, with a custom extractor reading its topic's title (Topic is
# not); the workshop holds the key to the topic, the row written, SET_NULL on
# delete.
DEPENDS_ON_FORWARD_FOREIGN_KEY_SET_NULL: _TargetCase[Topic, Workshop] = _TargetCase(
    register=_register_workshops_by_their_topic_title,
    create_target=_create_the_woodworking_topic,
    create_other_target=_whatever_the_target(_create_the_carving_topic),
    create_follower_on=_create_a_pottery_workshop,
    change_target=_retitle_the_topic,
    group_of=_group_of_a_pottery_workshop_by_its_topic_title,
    group_once_target_deleted=_group_of_a_pottery_workshop_by_its_nulled_topic,
)


def _register_intros_following_their_page() -> None:
    """Register PageIntro, following its page through its own one-to-one
    field: Page itself is not registered."""
    rag.register(PageIntro, follow=["page"])


def _create_an_intro_on(page: Page) -> PageIntro:
    """Create the given page's intro: it holds the one-to-one key to the
    page."""
    return PageIntro.objects.create(page=page, body="We build chairs by hand.")


def _group_of_an_intro_following_its_page(
    intro: PageIntro, /, *, target: Page
) -> dict[str, list[NormalizedDocument]]:
    """The intro's group: its body, then its page's title; its body is its
    title too."""
    return {
        f"testapp.pageintro:{intro.pk}": [
            NormalizedDocument(
                text=f"We build chairs by hand.\n\n{target.title}",
                source_app_label="testapp",
                source_model="pageintro",
                source_pk=intro.pk,
                title="We build chairs by hand.",
            ),
        ],
    }


def _group_of_an_intro_deleted_by_cascade(
    intro: PageIntro,
) -> dict[str, list[NormalizedDocument]]:
    """The intro's group once deleted with its page by cascade: empty."""
    return {f"testapp.pageintro:{intro.pk}": []}


# `follow` through a forward one-to-one: only PageIntro is registered,
# following its page (Page is not); the intro holds the one-to-one key to the
# page, the row written, CASCADE on delete. Each page has its own intro.
FOLLOW_FORWARD_ONE_TO_ONE: _TargetCase[Page, PageIntro] = _TargetCase(
    register=_register_intros_following_their_page,
    create_target=_create_the_about_page,
    create_other_target=_whatever_the_target(_create_the_workshop_page),
    create_follower_on=_create_an_intro_on,
    change_target=_retitle_the_page,
    group_of=_group_of_an_intro_following_its_page,
    group_once_target_deleted=_group_of_an_intro_deleted_by_cascade,
)


def _register_shelves_following_their_warehouse() -> None:
    """Register Shelf, following its warehouse through its own foreign key,
    which holds the Warehouse's code: Warehouse itself is not registered."""
    rag.register(Shelf, follow=["warehouse"])


def _create_a_warehouse_coded_by_the_pk_of(warehouse: Warehouse) -> Warehouse:
    """Create the South depot warehouse, coded by the given warehouse's primary
    key as text: a shelf matched by comparing that primary key with the stored
    code would be this one's, not the given warehouse's."""
    return Warehouse.objects.create(name="South depot", code=str(warehouse.pk))


def _rename_the_warehouse(warehouse: Warehouse) -> None:
    """Rename the given warehouse to North hall, without saving it."""
    warehouse.name = "North hall"


def _group_of_a_timber_shelf_following_its_warehouse(
    shelf: Shelf, /, *, target: Warehouse
) -> dict[str, list[NormalizedDocument]]:
    """The timber shelf's group: its label, then its warehouse's name; its
    label is its title too."""
    return {
        f"testapp.shelf:{shelf.pk}": [
            NormalizedDocument(
                text=f"Timber\n\n{target.name}",
                source_app_label="testapp",
                source_model="shelf",
                source_pk=shelf.pk,
                title="Timber",
            ),
        ],
    }


def _group_of_a_shelf_deleted_by_cascade(
    shelf: Shelf,
) -> dict[str, list[NormalizedDocument]]:
    """The shelf's group once deleted with its warehouse by cascade: empty."""
    return {f"testapp.shelf:{shelf.pk}": []}


def _recode_the_warehouse_moving_its_shelves(warehouse: Warehouse) -> None:
    """Change the given warehouse's code to north-hall and save it, then move
    its shelves to the new code."""
    old_code = warehouse.code
    # The code is the column the shelves point to the warehouse by: at the
    # save, the warehouse's shelf still holds the old code, so only the
    # warehouse as it was before the save tells which shelves named it.
    warehouse.code = "north-hall"
    warehouse.save()
    # The shelves are then moved to the new code, in the same transaction, by
    # an update that sends no signal: the foreign key constraint is checked at
    # the commit, and the shelf's own save is not observed.
    Shelf.objects.filter(warehouse_id=old_code).update(warehouse_id=warehouse.code)


# `follow` through a forward foreign key to a unique column: only Shelf is
# registered, following its warehouse (Warehouse is not); the shelf holds the
# key to the warehouse, the Warehouse's code rather than its primary key, the
# row written, CASCADE on delete. The other warehouse's code is the
# warehouse's primary key as text.
FOLLOW_FORWARD_FOREIGN_KEY_TO_FIELD: _TargetCase[Warehouse, Shelf] = _TargetCase(
    register=_register_shelves_following_their_warehouse,
    create_target=_create_the_north_warehouse,
    create_other_target=_create_a_warehouse_coded_by_the_pk_of,
    create_follower_on=_create_a_shelf_in,
    change_target=_rename_the_warehouse,
    group_of=_group_of_a_timber_shelf_following_its_warehouse,
    group_once_target_deleted=_group_of_a_shelf_deleted_by_cascade,
    change_key=_KeyChange(
        change=_recode_the_warehouse_moving_its_shelves,
        group_of=_group_of_a_timber_shelf_following_its_warehouse,
    ),
)


def _register_bins_following_their_depot() -> None:
    """Register Bin, following its depot through its own nullable foreign key,
    which holds the Depot's code: Depot itself is not registered."""
    rag.register(Bin, follow=["depot"])


def _create_the_north_depot() -> Depot:
    """Create the North depot, coded "north"."""
    return Depot.objects.create(name="North depot", code="north")


def _create_a_depot_coded_by_the_pk_of(depot: Depot) -> Depot:
    """Create the South depot, coded by the given depot's primary key as text:
    a bin matched by comparing that primary key with the stored code would be
    this one's, not the given depot's."""
    return Depot.objects.create(name="South depot", code=str(depot.pk))


def _create_a_spare_parts_bin_in(depot: Depot) -> Bin:
    """Create a Spare parts bin in the given depot: its foreign key holds the
    depot's code, not its primary key."""
    return Bin.objects.create(depot=depot, label="Spare parts")


def _rename_the_depot(depot: Depot) -> None:
    """Rename the given depot to North hall, without saving it."""
    depot.name = "North hall"


def _group_of_a_spare_parts_bin_following_its_depot(
    bin_: Bin, /, *, target: Depot
) -> dict[str, list[NormalizedDocument]]:
    """The spare parts bin's group: its label, then its depot's name; its
    label is its title too."""
    return {
        f"testapp.bin:{bin_.pk}": [
            NormalizedDocument(
                text=f"Spare parts\n\n{target.name}",
                source_app_label="testapp",
                source_model="bin",
                source_pk=bin_.pk,
                title="Spare parts",
            ),
        ],
    }


def _group_of_a_spare_parts_bin_whose_depot_is_nulled(
    bin_: Bin,
) -> dict[str, list[NormalizedDocument]]:
    """The spare parts bin's group once its depot is deleted and its foreign
    key set to null: the depot's name is gone, only its own label is left."""
    return {
        f"testapp.bin:{bin_.pk}": [
            NormalizedDocument(
                text="Spare parts",
                source_app_label="testapp",
                source_model="bin",
                source_pk=bin_.pk,
                title="Spare parts",
            ),
        ],
    }


def _recode_the_depot_moving_its_bins(depot: Depot) -> None:
    """Change the given depot's code to north-hall and save it, then move its
    bins to the new code."""
    old_code = depot.code
    # The depot is created with a code: its bins point to it by that code.
    assert old_code is not None
    # The code is the column the bins point to the depot by: at the save, the
    # depot's bin still holds the old code, so only the depot as it was before
    # the save tells which bins named it.
    depot.code = "north-hall"
    depot.save()
    # The bins are then moved to the new code, in the same transaction, by an
    # update that sends no signal: the foreign key constraint is checked at
    # the commit, and the bin's own save is not observed.
    Bin.objects.filter(depot_id=old_code).update(depot_id=depot.code)


# `follow` through a nullable forward foreign key to a unique column: only Bin
# is registered, following its depot (Depot is not); the bin holds the key to
# the depot, the Depot's code rather than its primary key, the row written,
# SET_NULL on delete. The other depot's code is the depot's primary key as
# text.
FOLLOW_FORWARD_FOREIGN_KEY_TO_FIELD_SET_NULL: _TargetCase[Depot, Bin] = _TargetCase(
    register=_register_bins_following_their_depot,
    create_target=_create_the_north_depot,
    create_other_target=_create_a_depot_coded_by_the_pk_of,
    create_follower_on=_create_a_spare_parts_bin_in,
    change_target=_rename_the_depot,
    group_of=_group_of_a_spare_parts_bin_following_its_depot,
    group_once_target_deleted=_group_of_a_spare_parts_bin_whose_depot_is_nulled,
    change_key=_KeyChange(
        change=_recode_the_depot_moving_its_bins,
        group_of=_group_of_a_spare_parts_bin_following_its_depot,
    ),
)


def _register_seminars_following_their_venue() -> None:
    """Register Seminar, by its title, following its venue through the
    multi-column ForeignObject ``venue``: Venue itself is not registered."""
    rag.register(Seminar, fields=["title"], follow=["venue"])


def _leave_the_venue_unchanged(_venue: Venue) -> None:
    """Change nothing on the given venue: Venue has no column outside the key
    the seminars name it by."""


def _group_of_a_seminar_following_its_venue(
    seminar: Seminar, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group: its title, then its venue's text fields, name
    first; its title is its document's title too."""
    return {
        f"testapp.seminar:{seminar.pk}": [
            NormalizedDocument(
                text=f"{_SEMINAR_TITLE}\n\n{target.name}\n\n{target.city}",
                source_app_label="testapp",
                source_model="seminar",
                source_pk=seminar.pk,
                title=_SEMINAR_TITLE,
            ),
        ],
    }


def _group_of_a_seminar_deleted_by_cascade(
    seminar: Seminar,
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group once deleted with its venue by cascade: empty."""
    return {f"testapp.seminar:{seminar.pk}": []}


def _rename_the_venue_leaving_its_seminars(venue: Venue) -> None:
    """Rename the given venue to Grande Halle and save it, its seminars left
    naming the old name."""
    # The name is one of the columns the seminars name the venue by: after the
    # save, the venue's seminar still carries the old name, so only the venue
    # as it was before the save tells which seminars named it.
    venue.name = "Grande Halle"
    venue.save()


def _group_of_a_seminar_whose_venue_was_renamed(
    seminar: Seminar, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group once its venue is renamed: its two columns now name
    no venue, so only its own title is left, not the stale venue text."""
    return {
        f"testapp.seminar:{seminar.pk}": [
            NormalizedDocument(
                text=_SEMINAR_TITLE,
                source_app_label="testapp",
                source_model="seminar",
                source_pk=seminar.pk,
                title=_SEMINAR_TITLE,
            ),
        ],
    }


# `follow` through a forward multi-column relation: only Seminar is
# registered, following its venue (Venue is not); the seminar holds the two
# columns naming the venue, its city and name, through the ForeignObject
# ``venue``, the row written, CASCADE on delete. The other venue is in the same
# city: only both columns together tell the venues apart.
FOLLOW_FORWARD_MULTI_COLUMN: _TargetCase[Venue, Seminar] = _TargetCase(
    register=_register_seminars_following_their_venue,
    create_target=_create_the_halle_tony_garnier,
    create_other_target=_whatever_the_target(_create_the_transbordeur),
    create_follower_on=_create_a_seminar_at,
    # Both columns are the key the seminars name the venue by: the save
    # changes neither, so the venue's seminars still follow it.
    change_target=_leave_the_venue_unchanged,
    group_of=_group_of_a_seminar_following_its_venue,
    group_once_target_deleted=_group_of_a_seminar_deleted_by_cascade,
    change_key=_KeyChange(
        change=_rename_the_venue_leaving_its_seminars,
        group_of=_group_of_a_seminar_whose_venue_was_renamed,
    ),
)


def _register_seminars_reading_their_venue_through_a_path() -> None:
    """Register Seminar, by its title, reading its venue's name through a
    lookup path across the multi-column ForeignObject ``venue``, with no
    follow: Venue itself is not registered."""
    rag.register(Seminar, fields=["title", "venue__name"])


def _group_of_a_seminar_reading_its_venue_name(
    seminar: Seminar, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group: its title, then its venue's name read through the
    path; its title is its document's title too."""
    return {
        f"testapp.seminar:{seminar.pk}": [
            NormalizedDocument(
                text=f"{_SEMINAR_TITLE}\n\n{target.name}",
                source_app_label="testapp",
                source_model="seminar",
                source_pk=seminar.pk,
                title=_SEMINAR_TITLE,
            ),
        ],
    }


# A lookup path through a forward multi-column relation: only Seminar is
# registered, reading its venue's name (Venue is not); the seminar holds the
# two columns naming the venue, its city and name, through the ForeignObject
# ``venue``, the row written, CASCADE on delete. The other venue is in the same
# city: only both columns together tell the venues apart. Once the venue is
# renamed, its seminar's two columns name no venue: the path reads nothing,
# only its own title is left.
PATH_FORWARD_MULTI_COLUMN: _TargetCase[Venue, Seminar] = _TargetCase(
    register=_register_seminars_reading_their_venue_through_a_path,
    create_target=_create_the_halle_tony_garnier,
    create_other_target=_whatever_the_target(_create_the_transbordeur),
    create_follower_on=_create_a_seminar_at,
    # Both columns are the key the seminars name the venue by: the save
    # changes neither, so the venue's seminars still read it.
    change_target=_leave_the_venue_unchanged,
    group_of=_group_of_a_seminar_reading_its_venue_name,
    group_once_target_deleted=_group_of_a_seminar_deleted_by_cascade,
    change_key=_KeyChange(
        change=_rename_the_venue_leaving_its_seminars,
        group_of=_group_of_a_seminar_whose_venue_was_renamed,
    ),
)


def _register_seminars_by_their_venue_name() -> None:
    """Register only Seminar, with a custom extractor reading its venue's name
    before its title when its two columns name a venue, its title alone
    otherwise: depends_on names the multi-column ForeignObject ``venue``, whose
    saves change its documents. Venue itself is not registered."""

    @rag.register_extractor(Seminar, depends_on=["venue"])
    class SeminarExtractor(BaseExtractor[Seminar]):
        def extract(self, instance: Seminar) -> NormalizedDocument:
            try:
                venue = instance.venue
            except Venue.DoesNotExist:
                return self.build_document(instance, text=instance.title)
            return self.build_document(instance, text=f"{venue.name}: {instance.title}")


def _group_of_a_seminar_by_its_venue_name(
    seminar: Seminar, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group, as its custom extractor builds it: its venue's
    name, then its title; no title in the document."""
    return {
        f"testapp.seminar:{seminar.pk}": [
            NormalizedDocument(
                text=f"{target.name}: {_SEMINAR_TITLE}",
                source_app_label="testapp",
                source_model="seminar",
                source_pk=seminar.pk,
            ),
        ],
    }


def _group_of_a_seminar_by_its_renamed_venue(
    seminar: Seminar, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The seminar's group, as its custom extractor builds it once its venue is
    renamed: its two columns now name no venue, so its title alone, not the
    stale venue name; no title in the document."""
    return {
        f"testapp.seminar:{seminar.pk}": [
            NormalizedDocument(
                text=_SEMINAR_TITLE,
                source_app_label="testapp",
                source_model="seminar",
                source_pk=seminar.pk,
            ),
        ],
    }


# `depends_on` through a forward multi-column relation: only Seminar is
# registered, with a custom extractor reading its venue's name (Venue is not);
# the seminar holds the two columns naming the venue, its city and name,
# through the ForeignObject ``venue``, the row written, CASCADE on delete. The
# other venue is in the same city: only both columns together tell the venues
# apart.
DEPENDS_ON_FORWARD_MULTI_COLUMN: _TargetCase[Venue, Seminar] = _TargetCase(
    register=_register_seminars_by_their_venue_name,
    create_target=_create_the_halle_tony_garnier,
    create_other_target=_whatever_the_target(_create_the_transbordeur),
    create_follower_on=_create_a_seminar_at,
    # Both columns are the key the seminars name the venue by: the save
    # changes neither, so the venue's seminars still depend on it.
    change_target=_leave_the_venue_unchanged,
    group_of=_group_of_a_seminar_by_its_venue_name,
    group_once_target_deleted=_group_of_a_seminar_deleted_by_cascade,
    change_key=_KeyChange(
        change=_rename_the_venue_leaving_its_seminars,
        group_of=_group_of_a_seminar_by_its_renamed_venue,
    ),
)


def _register_offers_reading_their_category_two_foreign_keys_deep() -> None:
    """Register Offer, reading its product's category's name through a lookup
    path two foreign keys deep, with no follow: neither Product nor Category is
    registered."""
    rag.register(Offer, fields=["title", "product__category__name"])


def _create_the_garden_category() -> Category:
    """Create the Garden category."""
    return Category.objects.create(name="Garden")


def _create_a_spring_sale_of_a_desk_lamp_in(category: Category) -> Offer:
    """Create a Desk lamp in the given category, then a Spring sale offer on
    it."""
    return Offer.objects.create(
        title="Spring sale", product=_create_a_plain_desk_lamp(category)
    )


def _group_of_a_spring_sale_reading_its_category(
    offer: Offer, /, *, target: Category
) -> dict[str, list[NormalizedDocument]]:
    """The spring sale's group: its title, then its product's category's
    name."""
    return {
        f"testapp.offer:{offer.pk}": [
            NormalizedDocument(
                text=f"Spring sale\n\n{target.name}",
                source_app_label="testapp",
                source_model="offer",
                source_pk=offer.pk,
                title="Spring sale",
            ),
        ],
    }


def _group_of_an_offer_deleted_by_cascade(
    offer: Offer,
) -> dict[str, list[NormalizedDocument]]:
    """The offer's group once deleted by cascade with its product, itself
    deleted with its category: empty."""
    return {f"testapp.offer:{offer.pk}": []}


# A lookup path two forward foreign keys deep: only Offer is registered,
# reading its product's category's name (neither Product nor Category is); the
# row written is the category, the last link of the path, Offer → product →
# category, both CASCADE on delete.
PATH_TWO_FORWARD_FOREIGN_KEYS: _TargetCase[Category, Offer] = _TargetCase(
    register=_register_offers_reading_their_category_two_foreign_keys_deep,
    create_target=_create_the_lighting_category,
    create_other_target=_whatever_the_target(_create_the_garden_category),
    create_follower_on=_create_a_spring_sale_of_a_desk_lamp_in,
    change_target=_rename_the_category,
    group_of=_group_of_a_spring_sale_reading_its_category,
    group_once_target_deleted=_group_of_an_offer_deleted_by_cascade,
)


def _register_sessions_reading_their_topic_two_links_deep() -> None:
    """Register Session, reading its workshop's topic's title through a lookup
    path two links deep, with no follow: neither Workshop nor Topic is
    registered."""
    rag.register(Session, fields=["title", "workshop__topic__title"])


def _create_a_morning_session_of_a_pottery_workshop_on(topic: Topic) -> Session:
    """Create a Pottery workshop on the given topic, then a Morning session of
    it."""
    return Session.objects.create(
        title="Morning", workshop=_create_a_pottery_workshop(topic)
    )


def _group_of_a_morning_session_reading_its_topic(
    session: Session, /, *, target: Topic
) -> dict[str, list[NormalizedDocument]]:
    """The morning session's group: its title, then its workshop's topic's
    title."""
    return {
        f"testapp.session:{session.pk}": [
            NormalizedDocument(
                text=f"Morning\n\n{target.title}",
                source_app_label="testapp",
                source_model="session",
                source_pk=session.pk,
                title="Morning",
            ),
        ],
    }


def _group_of_a_morning_session_whose_topic_is_nulled(
    session: Session,
) -> dict[str, list[NormalizedDocument]]:
    """The morning session's group once its workshop's topic is deleted and the
    workshop's foreign key set to null: the topic's title is gone, only the
    session's own title is left."""
    return {
        f"testapp.session:{session.pk}": [
            NormalizedDocument(
                text="Morning",
                source_app_label="testapp",
                source_model="session",
                source_pk=session.pk,
                title="Morning",
            ),
        ],
    }


# A lookup path two links deep, the last one nullable: only Session is
# registered, reading its workshop's topic's title (neither Workshop nor Topic
# is); the row written is the topic, the last link of the path, Session →
# workshop (CASCADE on delete) → topic (SET_NULL on delete). The delete sets
# the workshop's foreign key to null: the session is left, without the topic.
PATH_TWO_LINKS_SET_NULL: _TargetCase[Topic, Session] = _TargetCase(
    register=_register_sessions_reading_their_topic_two_links_deep,
    create_target=_create_the_woodworking_topic,
    create_other_target=_whatever_the_target(_create_the_carving_topic),
    create_follower_on=_create_a_morning_session_of_a_pottery_workshop_on,
    change_target=_retitle_the_topic,
    group_of=_group_of_a_morning_session_reading_its_topic,
    group_once_target_deleted=_group_of_a_morning_session_whose_topic_is_nulled,
)


def _register_talks_reading_their_venue_past_a_foreign_key() -> None:
    """Register Talk, reading its seminar's venue's name through a lookup path
    whose later link is the multi-column ForeignObject ``venue``, with no
    follow: neither Seminar nor Venue is registered."""
    rag.register(Talk, fields=["title", "seminar__venue__name"])


# The title a talk is created with.
_TALK_TITLE = "Keynote"


def _create_a_keynote_at(venue: Venue) -> Talk:
    """Create a seminar at the given venue, then a Keynote talk on it."""
    return Talk.objects.create(title=_TALK_TITLE, seminar=_create_a_seminar_at(venue))


def _group_of_a_keynote_reading_its_venue_name(
    talk: Talk, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The keynote's group: its title, then its seminar's venue's name read
    through the path; its title is its document's title too."""
    return {
        f"testapp.talk:{talk.pk}": [
            NormalizedDocument(
                text=f"{_TALK_TITLE}\n\n{target.name}",
                source_app_label="testapp",
                source_model="talk",
                source_pk=talk.pk,
                title=_TALK_TITLE,
            ),
        ],
    }


def _group_of_a_talk_deleted_by_cascade(
    talk: Talk,
) -> dict[str, list[NormalizedDocument]]:
    """The talk's group once deleted by cascade with its seminar, itself
    deleted with its venue: empty."""
    return {f"testapp.talk:{talk.pk}": []}


def _group_of_a_keynote_whose_venue_was_renamed(
    talk: Talk, /, *, target: Venue
) -> dict[str, list[NormalizedDocument]]:
    """The keynote's group once its seminar's venue is renamed: the seminar's
    two columns now name no venue, so only the talk's own title is left, not
    the stale venue name."""
    return {
        f"testapp.talk:{talk.pk}": [
            NormalizedDocument(
                text=_TALK_TITLE,
                source_app_label="testapp",
                source_model="talk",
                source_pk=talk.pk,
                title=_TALK_TITLE,
            ),
        ],
    }


# A lookup path past a forward foreign key then a multi-column relation: only
# Talk is registered, reading its seminar's venue's name (neither Seminar nor
# Venue is); the row written is the venue, the last link of the path, Talk →
# seminar by foreign key → venue by the multi-column ForeignObject, both
# CASCADE on delete. The other venue is in the same city: only both columns
# together tell the venues apart. Once the venue is renamed, its seminar's two
# columns name no venue: the path reads nothing, only the talk's own title is
# left.
PATH_FORWARD_FOREIGN_KEY_THEN_MULTI_COLUMN: _TargetCase[Venue, Talk] = _TargetCase(
    register=_register_talks_reading_their_venue_past_a_foreign_key,
    create_target=_create_the_halle_tony_garnier,
    create_other_target=_whatever_the_target(_create_the_transbordeur),
    create_follower_on=_create_a_keynote_at,
    # Both columns are the key the seminars name the venue by: the save
    # changes neither, so the talks of the venue's seminars still read it.
    change_target=_leave_the_venue_unchanged,
    group_of=_group_of_a_keynote_reading_its_venue_name,
    group_once_target_deleted=_group_of_a_talk_deleted_by_cascade,
    change_key=_KeyChange(
        change=_rename_the_venue_leaving_its_seminars,
        group_of=_group_of_a_keynote_whose_venue_was_renamed,
    ),
)


class _TargetWrite(Protocol):
    """A write of a target, generic over the case it is performed on."""

    def __call__(self, case: _TargetCase[TargetT, FollowerT], /) -> Act: ...


def _create_both_targets_with_their_followers(
    case: _TargetCase[TargetT, FollowerT],
) -> tuple[TargetT, FollowerT]:
    """Create the case's target with its follower, then the other target with a
    follower of its own, which the write leaves untouched; return the target
    and its follower."""
    target = case.create_target()
    follower = case.create_follower_on(target)
    case.create_follower_on(case.create_other_target(target))
    return target, follower


def _save_the_target(case: _TargetCase[TargetT, FollowerT]) -> Act:
    """The target changed and saved: one call, its follower's group."""
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the target's save is.
    target, follower = _create_both_targets_with_their_followers(case)

    def act() -> ReplaceCalls:
        case.change_target(target)
        target.save()
        return [case.group_of(follower, target=target)]

    return act


def _delete_the_target(case: _TargetCase[TargetT, FollowerT]) -> Act:
    """The target deleted: one call, its follower's group as the delete leaves
    it."""
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the target's delete is.
    target, follower = _create_both_targets_with_their_followers(case)

    def act() -> ReplaceCalls:
        target.delete()
        return [case.group_once_target_deleted(follower)]

    return act


def _change_the_target_key(case: _TargetCase[TargetT, FollowerT]) -> Act:
    """The target's key changed and saved, its followers then moved to the new
    key without a signal or left naming the old one: one call, its follower's
    group as committed."""
    change_key = case.change_key
    # Paired only with the cases that provide a key change.
    assert change_key is not None
    # Created before the write: the commit callbacks of these saves are not
    # observed, so only the target's key change is.
    target, follower = _create_both_targets_with_their_followers(case)

    def act() -> ReplaceCalls:
        change_key.change(target)
        return [change_key.group_of(follower, target=target)]

    return act


# Any: the cases pair different models, and _TargetCase is invariant in both,
# so no single precise type covers them all.
_TARGET_CASES: list[tuple[_TargetCase[Any, Any], str]] = [
    (FOLLOW_FORWARD_FOREIGN_KEY, "follow-forward_foreign_key"),
    (FOLLOW_FORWARD_FOREIGN_KEY_SET_NULL, "follow-forward_foreign_key_set_null"),
    (PATH_FORWARD_FOREIGN_KEY, "path-forward_foreign_key"),
    (PATH_FORWARD_FOREIGN_KEY_SET_NULL, "path-forward_foreign_key_set_null"),
    (
        PATH_FORWARD_FOREIGN_KEY_SET_NULL_AS_URL,
        "path-forward_foreign_key_set_null_as_url",
    ),
    (
        PATH_FORWARD_FOREIGN_KEY_SET_NULL_AS_TITLE,
        "path-forward_foreign_key_set_null_as_title",
    ),
    (DEPENDS_ON_FORWARD_FOREIGN_KEY, "depends_on-forward_foreign_key"),
    (
        DEPENDS_ON_FORWARD_FOREIGN_KEY_SET_NULL,
        "depends_on-forward_foreign_key_set_null",
    ),
    (FOLLOW_FORWARD_ONE_TO_ONE, "follow-forward_one_to_one"),
    (FOLLOW_FORWARD_FOREIGN_KEY_TO_FIELD, "follow-forward_foreign_key_to_field"),
    (
        FOLLOW_FORWARD_FOREIGN_KEY_TO_FIELD_SET_NULL,
        "follow-forward_foreign_key_to_field_set_null",
    ),
    (FOLLOW_FORWARD_MULTI_COLUMN, "follow-forward_multi_column"),
    (PATH_FORWARD_MULTI_COLUMN, "path-forward_multi_column"),
    (DEPENDS_ON_FORWARD_MULTI_COLUMN, "depends_on-forward_multi_column"),
    (PATH_TWO_FORWARD_FOREIGN_KEYS, "path-two_forward_foreign_keys"),
    (PATH_TWO_LINKS_SET_NULL, "path-two_links_set_null"),
    (
        PATH_FORWARD_FOREIGN_KEY_THEN_MULTI_COLUMN,
        "path-forward_foreign_key_then_multi_column",
    ),
]


def _target_writes_of(
    case: _TargetCase[TargetT, FollowerT],
) -> list[tuple[_TargetWrite, str]]:
    """The writes the given case is performed with: every case is saved and
    deleted; only a case providing a key change gets its key changed."""
    writes: list[tuple[_TargetWrite, str]] = [
        (_save_the_target, "save"),
        (_delete_the_target, "delete"),
    ]
    if case.change_key is not None:
        writes.append((_change_the_target_key, "change_key"))
    return writes


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("case", "write"),
    [
        pytest.param(case, write, id=f"{case_id}-{write_id}")
        for case, case_id in _TARGET_CASES
        for write, write_id in _target_writes_of(case)
    ],
)
def test_writing_a_row_its_followers_path_ends_on_replaces_the_followers_group(
    # Any: the cases pair different models, and _TargetCase is invariant in
    # both, so no single precise type covers them all.
    case: _TargetCase[Any, Any],
    write: _TargetWrite,
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The case registers its models; the write creates the rows it needs,
    # outside the captured callbacks.
    case.register()
    act = write(case)

    with django_capture_on_commit_callbacks(execute=True):
        expected = act()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Exactly the replace calls the write must send, in call order: none for
    # the other target's follower.
    assert _replaced(built_outputs) == expected


@pytest.mark.django_db
def test_deleting_a_notice_read_as_language_through_set_null_replaces_the_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Excerpt is registered, reading its language from its notice
    # through a lookup path over its nullable foreign key, SET_NULL on delete,
    # with no follow: Notice itself is not registered.
    _register_excerpts_reading_their_language_through_a_path()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the notice's delete below is observed.
    notice = _create_the_french_notice()
    greeting = _create_a_greeting_excerpt(notice)
    farewell = Excerpt.objects.create(title="Au revoir", notice=notice)
    # Another notice and its excerpt, untouched by the delete.
    other_notice = _create_the_english_notice()
    Excerpt.objects.create(title="Hello", notice=other_notice)

    # The delete sets the excerpts' foreign key to null before the notice's
    # row goes: by post_delete, the excerpts no longer point to the notice.
    with django_capture_on_commit_callbacks(execute=True):
        notice.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of the excerpts that pointed to the notice, as committed:
    # with no notice left, they have no language. Merged across replace
    # calls: whether they are sent in one call or several is not what this
    # test is about. No group of the other notice's excerpt.
    assert _received_groups(built_outputs) == {
        **_group_of_a_greeting_excerpt_whose_notice_is_nulled(greeting),
        f"testapp.excerpt:{farewell.pk}": [
            NormalizedDocument(
                text="Au revoir",
                source_app_label="testapp",
                source_model="excerpt",
                source_pk=farewell.pk,
                title="Au revoir",
                language=None,
            ),
        ],
    }


@pytest.mark.django_db
def test_deleting_a_bookmark_read_as_url_through_set_null_replaces_the_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Citation is registered, reading its url from its bookmark
    # through a lookup path over its nullable foreign key, SET_NULL on delete,
    # with no follow: Bookmark itself is not registered.
    _register_citations_reading_their_url_through_a_path()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the bookmark's delete below is observed.
    bookmark = _create_the_docs_bookmark()
    linked = _create_a_linked_citation(bookmark)
    quoted = Citation.objects.create(title="Quoted", bookmark=bookmark)
    # Another bookmark and its citation, untouched by the delete.
    other_bookmark = _create_the_example_bookmark()
    Citation.objects.create(title="Elsewhere", bookmark=other_bookmark)

    # The delete sets the citations' foreign key to null before the bookmark's
    # row goes: by post_delete, the citations no longer point to the bookmark.
    with django_capture_on_commit_callbacks(execute=True):
        bookmark.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of the citations that pointed to the bookmark, as committed:
    # with no bookmark left, they have no url, an empty string. Merged across
    # replace calls: whether they are sent in one call or several is not what
    # this test is about. No group of the other bookmark's citation.
    assert _received_groups(built_outputs) == {
        **_group_of_a_linked_citation_whose_bookmark_is_nulled(linked),
        f"testapp.citation:{quoted.pk}": [
            NormalizedDocument(
                text="Quoted",
                source_app_label="testapp",
                source_model="citation",
                source_pk=quoted.pk,
                title="Quoted",
                url="",
            ),
        ],
    }


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
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the product's save below is observed.
    lighting = _create_the_lighting_category()
    garden = _create_the_garden_category()
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
    lighting = _create_the_lighting_category()
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
    _register_offers_reading_their_category_two_foreign_keys_deep()

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
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's save below is observed.
    lighting = _create_the_lighting_category()
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
    _register_products_following_their_category()

    # More followers than SQLite accepts variables in one query. Created
    # outside the captured callbacks: bulk_create sends no signal anyway, so
    # only the category's save below is observed.
    lighting = _create_the_lighting_category()
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
    _register_products_following_their_category()

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
    _register_products_following_their_category()

    lighting = _create_the_lighting_category()
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
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's save below is observed.
    lighting = _create_the_lighting_category()
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
    lighting = _create_the_lighting_category()
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
    lighting = _create_the_lighting_category()
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
def test_saving_a_depot_with_a_null_code_replaces_no_group_of_the_bins_with_no_depot(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Bin is registered, following its depot through its own nullable
    # foreign key, which holds the Depot's code, not its primary key: Depot
    # itself is not.
    _register_bins_following_their_depot()

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
def test_deleting_a_depot_followed_through_set_null_replaces_the_groups_of_its_bins(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Bin is registered, following its depot through its own nullable
    # foreign key, which holds the Depot's code, not its primary key, SET_NULL
    # on delete: Depot itself is not.
    _register_bins_following_their_depot()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the depot's delete below is observed.
    north = _create_the_north_depot()
    spare_parts = _create_a_spare_parts_bin_in(north)
    fasteners = Bin.objects.create(depot=north, label="Fasteners")
    # Another depot and its bin, untouched by the delete.
    south = Depot.objects.create(name="South depot", code="south")
    Bin.objects.create(depot=south, label="Paint")

    # The delete sets the bins' foreign key to null before the depot's row
    # goes: by post_delete, the bins no longer hold the depot's code.
    with django_capture_on_commit_callbacks(execute=True):
        north.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of the bins that pointed to the depot, as committed: the
    # depot's name is gone, only each bin's own label is left. Merged across
    # replace calls: whether they are sent in one call or several is not what
    # this test is about. No group of the other depot's bin.
    assert _received_groups(built_outputs) == {
        f"testapp.bin:{spare_parts.pk}": [
            NormalizedDocument(
                text="Spare parts",
                source_app_label="testapp",
                source_model="bin",
                source_pk=spare_parts.pk,
                title="Spare parts",
            ),
        ],
        f"testapp.bin:{fasteners.pk}": [
            NormalizedDocument(
                text="Fasteners",
                source_app_label="testapp",
                source_model="bin",
                source_pk=fasteners.pk,
                title="Fasteners",
            ),
        ],
    }


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
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)
    tables = _create_a_plugin_on(about, "And tables too.")
    # A plugin of another page: the save below does not change its group.
    contact = Page.objects.create(title="Contact", slug="contact")
    _create_a_plugin_on(contact, "Write to us.")

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
    about = _create_the_about_page()

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
    chairs = _create_a_plugin_on(news)
    about = _create_the_about_page()

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
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)
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
def test_a_plugin_moved_by_update_after_its_own_save_replaces_both_pages_at_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The plugin starts on one page; the page it moves to has none.
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)
    workshop = _create_the_workshop_page()

    with django_capture_on_commit_callbacks(execute=True):
        chairs.body = "We build chairs and tables by hand."
        chairs.save()
        # Moved to the other page after its own save, in the same transaction,
        # by a write that sends no signal: only the save, still on its first
        # page, is observed.
        TextPlugin.objects.filter(pk=chairs.pk).update(page=workshop)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Both pages are replaced, as committed: the page the plugin
    # moved to with the plugin's text after its own title, and the page it left
    # with its title alone, the plugin's text gone.
    assert _received_groups(built_outputs) == {
        f"testapp.page:{workshop.pk}": [
            NormalizedDocument(
                text="Our workshop\n\nWe build chairs and tables by hand.",
                source_app_label="testapp",
                source_model="page",
                source_pk=workshop.pk,
                title="Our workshop",
                url="/pages/our-workshop/",
            ),
        ],
        f"testapp.page:{about.pk}": [
            NormalizedDocument(
                text="About us",
                source_app_label="testapp",
                source_model="page",
                source_pk=about.pk,
                title="About us",
                url="/pages/about-us/",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_plugin_created_then_moved_by_update_replaces_both_pages_at_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: their commit callbacks never
    # run. Neither page has a plugin yet.
    about = _create_the_about_page()
    workshop = _create_the_workshop_page()

    with django_capture_on_commit_callbacks(execute=True):
        chairs = TextPlugin.objects.create(page=about, body="We build chairs by hand.")
        # Moved to the other page after its creation, in the same transaction,
        # by a write that sends no signal: only the creation, on its first
        # page, is observed.
        TextPlugin.objects.filter(pk=chairs.pk).update(page=workshop)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Both pages are replaced, as committed: the page the plugin
    # moved to with the plugin's text after its own title, and the page it was
    # created on with its title alone, the plugin's text gone.
    assert _received_groups(built_outputs) == {
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
        f"testapp.page:{about.pk}": [
            NormalizedDocument(
                text="About us",
                source_app_label="testapp",
                source_model="page",
                source_pk=about.pk,
                title="About us",
                url="/pages/about-us/",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_product_moved_by_update_after_its_category_save_is_replaced_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    _register_products_following_their_category()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The product starts in another category; the category it moves to
    # has none.
    tools = _create_the_tools_category()
    lamp = _create_a_plain_desk_lamp(tools)
    lighting = _create_the_lighting_category()

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Moved to the category after the category's save, in the same
        # transaction, by a write that sends no signal: only the category's
        # save is observed.
        Product.objects.filter(pk=lamp.pk).update(category=lighting)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The product moved after the save is a follower of the
    # category at the commit, so its group is replaced, as committed: with the
    # new name of the category it moved to.
    assert _received_groups(built_outputs) == {
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


@pytest.mark.django_db
def test_a_product_moved_by_update_after_its_category_save_is_replaced_via_lookup_path(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, reading its category's name through a
    # lookup path, with no follow: Category itself is not registered.
    _register_products_reading_their_category_through_a_path()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The product starts in another category; the category it moves to
    # has none.
    tools = _create_the_tools_category()
    lamp = _create_a_plain_desk_lamp(tools)
    lighting = _create_the_lighting_category()

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Moved to the category after the category's save, in the same
        # transaction, by a write that sends no signal: only the category's
        # save is observed.
        Product.objects.filter(pk=lamp.pk).update(category=lighting)

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The product moved after the save reads the category
    # through the path at the commit, so its group is replaced, as committed:
    # with the new name of the category it moved to.
    assert _received_groups(built_outputs) == {
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
    lighting = _create_the_lighting_category()
    garden = _create_the_garden_category()
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
    lighting = _create_the_lighting_category()
    garden = _create_the_garden_category()
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
    lighting = _create_the_lighting_category()
    garden = _create_the_garden_category()
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

    _register_topics_by_their_course_titles()

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

    _register_topics_by_their_course_titles()

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

    _register_topics_by_their_course_titles()

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
def test_removing_a_topic_from_a_course_its_custom_extractor_depends_on_replaces_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_topics_by_their_course_titles()

    # Created and linked outside the captured callbacks: the commit callbacks
    # of these saves and adds never run, so only the remove below is observed.
    # The topic is covered by two courses, so that its group keeps the one
    # left.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    joinery = Course.objects.create(title="Joinery")
    basics.topics.add(woodworking)
    joinery.topics.add(woodworking)

    # Removed from the course's side: Django sends m2m_changed with the course
    # as its instance, and the topic among the primary keys it names. Neither
    # row is saved again: the remove deletes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.remove(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Topic's group as committed: the removed course's
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
def test_clearing_the_topics_of_a_course_custom_extractor_depends_on_replaces_each(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_topics_by_their_course_titles()

    # Created and linked outside the captured callbacks: the commit callbacks
    # of these saves and adds never run, so only the clear below is observed.
    # The course covers two topics, so that the clear removes more than one
    # link; one of them is covered by another course too, so that its group
    # keeps that course.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)
    whittling = Course.objects.create(title="Whittling")
    whittling.topics.add(carving)

    # Cleared from the course's side: Django sends m2m_changed with the course
    # as its instance, and no primary keys at all. Neither row is saved again:
    # the clear deletes only the links of the course.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The groups of both topics as committed, each without the
    # cleared course's title: the one covered by another course keeps that
    # course's title.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking: ",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
            ),
        ],
        f"testapp.topic:{carving.pk}": [
            NormalizedDocument(
                text="Carving: Whittling",
                source_app_label="testapp",
                source_model="topic",
                source_pk=carving.pk,
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
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)
    blank = _create_a_plugin_on(about, "")

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
    _register_products_following_their_category()

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
    _register_products_following_their_category()

    lighting = _create_the_lighting_category()
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
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the categories' saves below are observed.
    lighting = _create_the_lighting_category()
    # Two followers, so that the message cannot name the one follower there is.
    lamp = _create_a_plain_desk_lamp(lighting)
    _create_a_bulb(lighting)
    tools = _create_the_tools_category()
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
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's save below is observed.
    lighting = _create_the_lighting_category()
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
    _register_workshops_following_their_topic()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = _create_a_pottery_workshop(woodworking)
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
def test_a_topic_saved_then_deleted_replaces_no_group_of_the_workshops_with_no_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    _register_workshops_following_their_topic()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's save and delete below are observed.
    woodworking = _create_the_woodworking_topic()
    pottery = _create_a_pottery_workshop(woodworking)
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
def test_deleting_a_category_whose_following_products_cascade_sends_only_their_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key, CASCADE on delete: Category itself is not.
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the category's delete below is observed.
    lighting = _create_the_lighting_category()
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
    _register_workshops_following_their_topic()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the proxy's delete below is observed.
    woodworking = _create_the_woodworking_topic()
    pottery = _create_a_pottery_workshop(woodworking)

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
    pottery = _create_a_pottery_workshop(woodworking)

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    _register_workshops_following_their_topic()

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
    page = _create_the_about_page()

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
def test_moving_a_followed_instance_built_with_an_existing_pk_replaces_both_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the plugin's move below is observed.
    about = _create_the_about_page()
    workshop = _create_the_workshop_page()
    existing_plugin = _create_a_plugin_on(about)
    _create_a_plugin_on(about, "We ship worldwide.")

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
    lighting = _create_the_lighting_category()
    tools = _create_the_tools_category()
    garden = _create_the_garden_category()
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
def test_a_product_saved_then_moved_by_update_skips_the_category_it_only_passed_by(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # A Product is followed by its Category, through the reverse relation
    # ``products``. Product itself is not registered.
    rag.register(Category, follow=["products"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the product's moves below are observed.
    lighting = _create_the_lighting_category()
    tools = _create_the_tools_category()
    garden = _create_the_garden_category()
    lamp = _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lamp.category = tools
        lamp.save()
        # Moved again after the save, in the same transaction, by a write that
        # sends no signal: the Tools category never has the product in any
        # committed state.
        Product.objects.filter(pk=lamp.pk).update(category=garden)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The groups of the category the product left and of the
    # one it ends in, both as committed; no group of the category it only
    # passed by before the commit.
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
        f"testapp.category:{garden.pk}": [
            NormalizedDocument(
                text="Garden\n\nDesk lamp\n\nA lamp for the desk.\n\nNew",
                source_app_label="testapp",
                source_model="category",
                source_pk=garden.pk,
                title="Garden",
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
    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)

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
    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)

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
def test_moving_a_child_with_a_primary_key_of_its_own_replaces_both_categories(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Neither Product nor ClearanceProduct is registered: a Category follows
    # its products through the reverse relation ``products``.
    rag.register(Category, follow=["products"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the clearance product's move below is observed.
    lighting = _create_the_lighting_category()
    tools = _create_the_tools_category()
    # The child's primary key is its code, not the Product's: its Product row
    # is reached by the explicit parent link, ``product``, whose value is the
    # Product's integer primary key, never the code.
    lamp = _create_a_clearance_desk_lamp(lighting)

    # Django sends pre_save and post_save with ClearanceProduct as their
    # sender, not Product: the category the Product row had before the save
    # is found from the parent link, not from the code.
    with django_capture_on_commit_callbacks(execute=True):
        lamp.category = tools
        lamp.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Both categories' groups as committed: the old one loses
    # the product's text, the new one gains it.
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
    }


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
    tools = _create_the_tools_category()

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
def test_updating_a_followed_instance_linked_by_a_unique_column_reads_it_at_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_warehouses_following_their_shelves()

    # Created outside the counted queries: only the shelf's update below is
    # observed.
    north = _create_the_north_warehouse()
    shelf = _create_a_shelf_in(north)

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits: the lookup of the Warehouse before the save,
    # the save's UPDATE, the lookup of the Warehouse from the row as
    # committed, then the three reads of the Warehouse's group. The Warehouse
    # is not also read at post_save by the code the shelf holds in memory: the
    # lookup at the commit already finds it.
    with (
        django_assert_num_queries(6) as queries,
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Saved again in place: the shelf's foreign key still holds the
        # Warehouse's code, "north", not its primary key.
        shelf.label = "Paint"
        shelf.save()

    # No query turns the code the shelf holds into the Warehouse's primary key.
    by_code = 'FROM "testapp_warehouse" WHERE "testapp_warehouse"."code"'
    assert not [query for query in queries.captured_queries if by_code in query["sql"]]
    # The Warehouse's group, once, under its primary key, not its code.
    assert _replaced(built_outputs) == [
        {
            f"testapp.warehouse:{north.pk}": [
                NormalizedDocument(
                    text="North depot\n\nPaint",
                    source_app_label="testapp",
                    source_model="warehouse",
                    source_pk=north.pk,
                    title="North depot",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_updating_a_match_followed_by_two_reverse_relations_looks_teams_up_once_each(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Team is registered.
    _register_teams_following_their_matches()

    # Created outside the counted queries: only the match's update below is
    # observed.
    lyon = _create_lyon()
    nantes = _create_nantes()
    derby = _create_a_match(home_team=lyon, away_team=nantes)

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits: one lookup of the Teams before the save, the
    # save's UPDATE, one lookup of the Teams from the row as committed, then
    # the four reads of the Teams' groups. Each lookup crosses both reverse
    # relations at once, not one query per relation: two queries fewer than
    # one lookup per relation at each moment.
    with (
        django_assert_num_queries(7),
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Saved again in place: the match keeps its home and away teams.
        derby.title = "Season opener"
        derby.save()

    # The groups replaced do not change: each Team's group, once, in one call,
    # with the match's title as committed after the Team's own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.team:{lyon.pk}": [
                NormalizedDocument(
                    text="Lyon\n\nSeason opener",
                    source_app_label="testapp",
                    source_model="team",
                    source_pk=lyon.pk,
                    title="Lyon",
                ),
            ],
            f"testapp.team:{nantes.pk}": [
                NormalizedDocument(
                    text="Nantes\n\nSeason opener",
                    source_app_label="testapp",
                    source_model="team",
                    source_pk=nantes.pk,
                    title="Nantes",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_updating_a_match_followed_by_two_registered_models_looks_each_up_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Two registered models follow Match in reverse: the Team by both its
    # reverse foreign keys, ``home_matches`` and ``away_matches``, and the
    # Tournament by its own, ``matches``. Match itself is not registered.
    _register_teams_following_their_matches()
    rag.register(Tournament, fields=["name"], follow=["matches"])

    # Created outside the counted queries: only the match's update below is
    # observed.
    lyon = _create_lyon()
    nantes = _create_nantes()
    cup = Tournament.objects.create(name="Spring Cup")
    derby = _create_a_match(home_team=lyon, away_team=nantes, tournament=cup)

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits: one lookup per registered model before the
    # save (two), the save's UPDATE, one lookup per registered model from the
    # row as committed (two), then the reads of the groups: the Teams' four
    # and the Tournament's three. Three reverse relations, but two registered
    # models: the lookups grow with the models, not with the relations.
    with (
        django_assert_num_queries(12),
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Saved again in place: the match keeps its teams and its tournament.
        derby.title = "Season opener"
        derby.save()

    # Each following row's group, with the match's title as committed after
    # the row's own name, merged across replace calls: whether the groups come
    # in one call or one per registered model is not what this test is about.
    assert _received_groups(built_outputs) == {
        f"testapp.team:{lyon.pk}": [
            NormalizedDocument(
                text="Lyon\n\nSeason opener",
                source_app_label="testapp",
                source_model="team",
                source_pk=lyon.pk,
                title="Lyon",
            ),
        ],
        f"testapp.team:{nantes.pk}": [
            NormalizedDocument(
                text="Nantes\n\nSeason opener",
                source_app_label="testapp",
                source_model="team",
                source_pk=nantes.pk,
                title="Nantes",
            ),
        ],
        f"testapp.tournament:{cup.pk}": [
            NormalizedDocument(
                text="Spring Cup\n\nSeason opener",
                source_app_label="testapp",
                source_model="tournament",
                source_pk=cup.pk,
                title="Spring Cup",
            ),
        ],
    }
    # Each group sent once: no following row is replaced twice.
    sent_source_keys = [
        source_key for groups in _replaced(built_outputs) for source_key in groups
    ]
    assert sorted(sent_source_keys) == sorted(
        [
            f"testapp.team:{lyon.pk}",
            f"testapp.team:{nantes.pk}",
            f"testapp.tournament:{cup.pk}",
        ]
    )


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

    _register_topics_following_their_courses()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the remove below is observed. The
    # course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed.
    # The course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the clear below is observed. The
    # course covers two topics, so that the clear removes more than one link.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Two courses cover the topic: one covers another topic too, so that its
    # group keeps the one left, the other covers it alone.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # A topic the course already covers: the add below does not change its
    # group.
    carving = _create_the_carving_topic()
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

    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    # Another topic the course already covers: the add below does not change
    # its group.
    carving = _create_the_carving_topic()
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

    # TopicProxy is not registered either.
    _register_topics_following_their_courses()

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

    # TopicProxy is not registered either.
    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Two courses cover the topic: one covers another topic too, so that its
    # group keeps the one left, the other covers it alone.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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
def test_clearing_the_topics_of_a_course_child_replaces_the_group_of_each_topic_it_held(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Neither Course nor MasterClass is registered.
    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed. The
    # master class covers two topics, so that the clear removes more than one
    # link; one of them is covered by another course too, so that its group
    # keeps that course.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
    masterclass = MasterClass.objects.create(
        title="Woodworking masterclass", instructor="Ada"
    )
    masterclass.topics.add(woodworking, carving)
    whittling = Course.objects.create(title="Whittling")
    whittling.topics.add(carving)

    # Cleared from the child's side: Django sends m2m_changed with the
    # MasterClass as its instance, not a Course, while the links name its
    # Course row, and no primary keys at all. No row is saved again: the clear
    # deletes only the links of the master class.
    with django_capture_on_commit_callbacks(execute=True):
        masterclass.topics.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of both topics as committed, each without the master class's
    # title: the one covered by another course keeps that course's title.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking\n\nJoints and finishes.",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
                title="Woodworking",
            ),
        ],
        f"testapp.topic:{carving.pk}": [
            NormalizedDocument(
                text="Carving\n\nKnives and gouges.\n\nWhittling",
                source_app_label="testapp",
                source_model="topic",
                source_pk=carving.pk,
                title="Carving",
            ),
        ],
    }


class _FailedClearError(Exception):
    """Raised by a receiver of pre_clear, to make a clear fail before its DELETE."""


@pytest.mark.django_db
def test_adding_a_topic_after_a_failed_clear_replaces_only_the_added_topics_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed. The
    # course covers one topic, the one the failed clear reads; the other is
    # the one the add below links.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)

    def fail_on_pre_clear(action: str, **kwargs: Any) -> None:
        if action == "pre_clear":
            raise _FailedClearError

    # Connected after the package's receiver, connected at app ready: that one
    # reads the keys of the links before this one makes the clear fail, so
    # post_clear never comes.
    m2m_changed.connect(fail_on_pre_clear, sender=Course.topics.through)
    try:
        # A savepoint of its own, so that the failure rolls back only the clear.
        with pytest.raises(_FailedClearError), transaction.atomic():
            basics.topics.clear()
    finally:
        m2m_changed.disconnect(fail_on_pre_clear, sender=Course.topics.through)

    # Added from the course's side: Django sends m2m_changed with the same
    # course instance the failed clear had, and only the added topic among the
    # primary keys it names.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(carving)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The added Topic's group as committed, and no group of the topic the
    # failed clear read: the course still covers it, unchanged.
    assert _replaced(built_outputs) == [
        {
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
    ]


@pytest.mark.django_db
def test_adding_two_topics_to_a_course_replaces_the_group_of_each_topic_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_guilds_following_their_members()

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
def test_removing_a_guild_from_a_craftsman_replaces_the_guilds_group_named_by_its_code(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_guilds_following_their_members()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed. A
    # guild's code, a slug, can never equal its integer primary key. The guild
    # has two members, so that its group keeps the one left.
    carpenter = Craftsman.objects.create(name="Carpenter")
    mason = Craftsman.objects.create(name="Mason")
    north = Guild.objects.create(name="North guild", code="north")
    north.members.add(carpenter, mason)
    # A guild the craftsman still belongs to after the remove below: its group
    # does not change.
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(carpenter)

    # Removed from the craftsman's side: Django sends m2m_changed with the
    # craftsman as its instance, and names the guild by the column
    # Membership.guild points to, its code, not by its primary key. Neither row
    # is saved again: the remove deletes only the membership between them.
    with django_capture_on_commit_callbacks(execute=True):
        carpenter.guilds.remove(north)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The removed Guild's group as committed: the craftsman's name is gone,
    # only the Guild's own name and its other member's name are left. No group
    # of the guild the craftsman still belongs to.
    assert _replaced(built_outputs) == [
        {
            f"testapp.guild:{north.pk}": [
                NormalizedDocument(
                    text="North guild\n\nMason",
                    source_app_label="testapp",
                    source_model="guild",
                    source_pk=north.pk,
                    title="North guild",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_clearing_the_members_of_a_guild_replaces_the_group_of_each_craftsman_in_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Craftsman is registered, following its guilds through the
    # reverse many-to-many ``guilds``: Guild itself is not.
    rag.register(Craftsman, fields=["name"], follow=["guilds"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed. A
    # guild's code, a slug, can never equal its integer primary key. The
    # craftsman belongs to another guild too, so that its group keeps the one
    # left.
    carpenter = Craftsman.objects.create(name="Carpenter")
    north = Guild.objects.create(name="North guild", code="north")
    south = Guild.objects.create(name="South guild", code="south")
    north.members.add(carpenter)
    south.members.add(carpenter)
    # A craftsman not in the guild: the clear below does not change its group.
    mason = Craftsman.objects.create(name="Mason")
    south.members.add(mason)

    # Cleared from the guild's side: Django sends m2m_changed with the guild as
    # its instance and no primary keys at all, so its members can only be found
    # before the clear, by the memberships naming the guild by the column
    # Membership.guild points to, its code, not by its primary key. No row is
    # saved again: the clear deletes only the memberships of the guild.
    with django_capture_on_commit_callbacks(execute=True):
        north.members.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Craftsman's group as committed: its name, then the name of the guild
    # it still belongs to. No group of the craftsman never in the guild.
    assert _replaced(built_outputs) == [
        {
            f"testapp.craftsman:{carpenter.pk}": [
                NormalizedDocument(
                    text="Carpenter\n\nSouth guild",
                    source_app_label="testapp",
                    source_model="craftsman",
                    source_pk=carpenter.pk,
                    title="Carpenter",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_musician_to_a_band_replaces_the_musicians_group_named_by_its_handle(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and this add never run, so only the add below is observed. A
    # musician's handle, a slug, can never equal its integer primary key.
    quartet = Band.objects.create(name="Quartet")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician already in the band: the add below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ben)

    # Added from the band's side: Django sends m2m_changed with the band as its
    # instance, and names the musician by the column Engagement.musician points
    # to, its handle, not by its primary key. Neither row is saved again: the
    # add writes only the engagement between them.
    with django_capture_on_commit_callbacks(execute=True):
        quartet.musicians.add(ada)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The added Musician's group as committed: its name, then its new band's
    # name. No group of the musician already in the band.
    assert _replaced(built_outputs) == [
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nQuartet",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_removing_a_musician_from_a_band_replaces_its_group_named_by_its_handle(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # removed musician plays in another band too, so that its group keeps that
    # band.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician left in the band: the remove below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada)

    # Removed from the band's side: Django sends m2m_changed with the band as
    # its instance, and names the musician by the column Engagement.musician
    # points to, its handle, not by its primary key. Neither row is saved
    # again: the remove deletes only the engagement between them.
    with django_capture_on_commit_callbacks(execute=True):
        quartet.musicians.remove(ada)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The removed Musician's group as committed: the band's name is gone, only
    # its own name and its other band's name are left. No group of the
    # musician left in the band.
    assert _replaced(built_outputs) == [
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nTrio",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_removing_a_band_from_a_musician_replaces_the_musicians_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # musician plays in another band too, so that its group keeps that band.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician left in the band: the remove below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada)

    # Removed from the musician's side: Django sends m2m_changed with the
    # musician as its instance and the band's primary key in pk_set, so the
    # musician to replace is the instance itself, not a row named in pk_set.
    # Neither row is saved again: the remove deletes only the engagement
    # between them.
    with django_capture_on_commit_callbacks(execute=True):
        ada.bands.remove(quartet)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Musician's group as committed: the removed band's name is gone, only
    # its own name and its other band's name are left. No group of the
    # musician left in the band.
    assert _replaced(built_outputs) == [
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nTrio",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_adding_a_band_to_a_musician_replaces_only_the_musicians_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the add below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # musician already plays in the quartet, so that its group keeps that band.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    quartet.musicians.add(ada)
    # A musician already in the trio: the add below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    trio.musicians.add(ben)

    # Added from the musician's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with the musician as its instance and the band's
    # primary key in pk_set, so the musician to replace is the instance itself,
    # not a row named in pk_set. Neither row is saved again: the add writes
    # only the engagement between them.
    with django_capture_on_commit_callbacks(execute=True):
        ada.bands.add(trio)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Musician's group as committed: its own name, then the
    # names of the band it already played in and the band it was added to. No
    # group of the musician already in the trio, nor of the unregistered band.
    assert _received_groups(built_outputs) == {
        f"testapp.musician:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nQuartet\n\nTrio",
                source_app_label="testapp",
                source_model="musician",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_bands_of_a_musician_replaces_only_the_musicians_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # musician plays in two bands, so that the clear removes more than one link.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician of each of its bands: the clear below changes neither group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    cleo = Musician.objects.create(name="Cleo", handle="cleo")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada, cleo)

    # Cleared from the musician's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with the musician as its instance and no primary
    # keys at all, so the musician to replace is the instance itself, not a row
    # found through its former bands. No row is saved again: the clear deletes
    # only the engagements of the musician.
    with django_capture_on_commit_callbacks(execute=True):
        ada.bands.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The Musician's group as committed: both bands' names are
    # gone, only its own name is left. No group of the other musicians of its
    # former bands, nor of the unregistered bands.
    assert _received_groups(built_outputs) == {
        f"testapp.musician:{ada.pk}": [
            NormalizedDocument(
                text="Ada",
                source_app_label="testapp",
                source_model="musician",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_musicians_of_a_band_replaces_the_group_of_each_musician_in_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # band holds two musicians, so that the clear removes more than one link;
    # one of them plays in another band too, so that its group keeps that band.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada)
    # A musician not in the band: the clear below does not change its group.
    cleo = Musician.objects.create(name="Cleo", handle="cleo")
    trio.musicians.add(cleo)

    # Cleared from the band's side: Django sends m2m_changed with the band as
    # its instance and no primary keys at all, so its musicians can only be
    # found before the clear, by the engagements naming them by the column
    # Engagement.musician points to, their handle, not by their primary key. No
    # row is saved again: the clear deletes only the engagements of the band.
    with django_capture_on_commit_callbacks(execute=True):
        quartet.musicians.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The groups of both musicians as committed, each without
    # the cleared band's name: the one playing in another band keeps that
    # band's name. No group of the musician never in the band.
    assert _received_groups(built_outputs) == {
        f"testapp.musician:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nTrio",
                source_app_label="testapp",
                source_model="musician",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
        f"testapp.musician:{ben.pk}": [
            NormalizedDocument(
                text="Ben",
                source_app_label="testapp",
                source_model="musician",
                source_pk=ben.pk,
                title="Ben",
            ),
        ],
    }


@pytest.mark.django_db
def test_setting_the_musicians_of_a_band_replaces_the_groups_of_those_removed_and_added(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the set below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # musician the set removes plays in another band too, so that its group
    # keeps that band.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician kept in the band: the set below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada)
    # A musician not in the band yet: the set below adds it to the band.
    cleo = Musician.objects.create(name="Cleo", handle="cleo")

    # Set from the band's side: Django compares the musicians named by the
    # column Engagement.musician points to, their handle, then sends
    # m2m_changed twice with the band as its instance, a remove naming Ada by
    # its handle and an add naming Cleo by its own, never Ben. No row is saved
    # again: the set writes only the engagements of the band.
    with django_capture_on_commit_callbacks(execute=True):
        quartet.musicians.set([ben, cleo])
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # set() sends a remove then an add, each replacing its own groups, so the
    # calls are pinned exactly: batching them into one call is a later step's
    # work, which will change this expected value. The removed musician's
    # group without the band's name, keeping its other band's; the added
    # musician's group with it. No group of the musician kept in the band.
    assert _replaced(built_outputs) == [
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nTrio",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        },
        {
            f"testapp.musician:{cleo.pk}": [
                NormalizedDocument(
                    text="Cleo\n\nQuartet",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=cleo.pk,
                    title="Cleo",
                ),
            ],
        },
    ]


@pytest.mark.django_db
def test_setting_the_bands_of_a_musician_replaces_the_musicians_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_musicians_following_their_bands()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the set below is observed. A
    # musician's handle, a slug, can never equal its integer primary key. The
    # musician plays in the quartet and the trio: the set below removes it from
    # the quartet, keeps it in the trio and adds it to the duo.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    duo = Band.objects.create(name="Duo")
    ada = Musician.objects.create(name="Ada", handle="ada")
    # A musician left in the quartet: the set below does not change its group.
    ben = Musician.objects.create(name="Ben", handle="ben")
    quartet.musicians.add(ada, ben)
    trio.musicians.add(ada)

    # Set from the musician's side: Django compares the bands by their primary
    # key, then sends m2m_changed twice with the musician as its instance, a
    # remove with the quartet's primary key in pk_set and an add with the duo's,
    # never the trio's. The musician to replace is the instance itself, not a
    # row named in pk_set. No row is saved again: the set writes only the
    # engagements of the musician.
    with django_capture_on_commit_callbacks(execute=True):
        ada.bands.set([trio, duo])
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # set() sends a remove then an add, each replacing its own groups, so the
    # calls are pinned exactly: the musician's group is replaced twice,
    # identically. Batching them into one call is a later step's work, which
    # will change this expected value. The musician's group as committed: the
    # quartet's name is gone, its own name, then the names of the band it was
    # kept in and the band it was added to are left. No group of the musician
    # left in the quartet.
    assert _replaced(built_outputs) == [
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nTrio\n\nDuo",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        },
        {
            f"testapp.musician:{ada.pk}": [
                NormalizedDocument(
                    text="Ada\n\nTrio\n\nDuo",
                    source_app_label="testapp",
                    source_model="musician",
                    source_pk=ada.pk,
                    title="Ada",
                ),
            ],
        },
    ]


@pytest.mark.django_db
def test_adding_a_mentor_to_a_person_replaces_the_mentors_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the add below is observed. Grace
    # mentors another person already, so that its group shows that mentee
    # next to the one added. Alan is created and linked first, so that its
    # name comes before Ada's whichever order the mentees are read in.
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    alan.mentors.add(grace)
    ada = Person.objects.create(name="Ada")
    # A person not mentoring Ada: the add below does not change Barbara's
    # group.
    barbara = Person.objects.create(name="Barbara")
    alan.mentors.add(barbara)

    # Added from the mentee's side: Django sends m2m_changed with Ada as its
    # instance and Grace's primary key in pk_set, so the person to replace is
    # the one named in pk_set, the mentor, not the instance. No row is saved
    # again: the add writes only Ada's link to its mentor.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.add(grace)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Grace's group as committed, with Ada's name added after its other
    # mentee's, and no group of the person not mentoring Ada. Ada's own
    # group is not replaced: it follows its mentees, not its mentors, so the
    # add does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace\n\nAlan\n\nAda",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_mentor_to_a_person_following_its_mentors_replaces_only_its_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentors()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the add below is observed. Ada
    # has a mentor already, so that its group shows that mentor next to the
    # one added. Linus is created and linked first, so that its name comes
    # before Grace's whichever order the mentors are read in.
    ada = Person.objects.create(name="Ada")
    linus = Person.objects.create(name="Linus")
    ada.mentors.add(linus)
    grace = Person.objects.create(name="Grace")
    # A person not mentoring Ada, mentoring Grace: Grace's group lists its own
    # mentors, which the add below does not change, and neither does it
    # change Barbara's group.
    barbara = Person.objects.create(name="Barbara")
    grace.mentors.add(barbara)

    # Added from the mentee's side, the forward side of the many-to-many:
    # Django sends m2m_changed with Ada as its instance and Grace's primary
    # key in pk_set, so the person to replace is the instance, the mentee,
    # not the one named in pk_set. No row is saved again: the add writes only
    # Ada's link to its new mentor.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.add(grace)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Ada's group as committed, with Grace's name added after its other
    # mentor's, and no group of the person not mentoring Ada. Grace's own
    # group is not replaced: it follows its mentors, not its mentees, so the
    # add does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nLinus\n\nGrace",
                source_app_label="testapp",
                source_model="person",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
    }


@pytest.mark.django_db
def test_removing_a_mentor_from_a_person_replaces_the_mentors_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed.
    # Ada has two mentors: Grace, which mentors another person too, so that its
    # group keeps the mentee left, and Linus, which Ada keeps.
    ada = Person.objects.create(name="Ada")
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    linus = Person.objects.create(name="Linus")
    ada.mentors.add(grace, linus)
    alan.mentors.add(grace)

    # Removed from the mentee's side: Django sends m2m_changed with Ada as its
    # instance and Grace's primary key in pk_set, so the person to replace is
    # the one named in pk_set, the mentor, not the instance. No row is saved
    # again: the remove deletes only Ada's link to Grace.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.remove(grace)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Grace's group as committed, without Ada's name, its other mentee kept,
    # and no group of the mentor Ada keeps. Ada's own group is not replaced:
    # it follows its mentees, not its mentors, so the remove does not change
    # it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace\n\nAlan",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_mentee_to_a_mentor_replaces_the_mentors_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the add below is observed. Grace
    # mentors another person already, so that its group shows that mentee
    # next to the one added. Alan is created and linked first, so that its
    # name comes before Ada's whichever order the mentees are read in.
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    grace.mentees.add(alan)
    ada = Person.objects.create(name="Ada")
    # A person not mentoring Ada: the add below does not change Barbara's
    # group.
    barbara = Person.objects.create(name="Barbara")
    barbara.mentees.add(alan)

    # Added from the mentor's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with Grace as its instance and Ada's primary
    # key in pk_set, so the person to replace is the instance, the mentor,
    # not the one named in pk_set. No row is saved again: the add writes only
    # Grace's link to its new mentee.
    with django_capture_on_commit_callbacks(execute=True):
        grace.mentees.add(ada)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Grace's group as committed, with Ada's name added after its other
    # mentee's, and no group of the person not mentoring Ada. Ada's own
    # group is not replaced: it follows its mentees, not its mentors, so the
    # add does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace\n\nAlan\n\nAda",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_mentee_to_a_mentor_following_mentors_replaces_only_the_mentees_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentors()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the add below is observed. Ada
    # has a mentor already, so that its group shows that mentor next to the
    # one added. Linus is created and linked first, so that its name comes
    # before Grace's whichever order the mentors are read in.
    ada = Person.objects.create(name="Ada")
    linus = Person.objects.create(name="Linus")
    linus.mentees.add(ada)
    grace = Person.objects.create(name="Grace")
    # A person not mentoring Ada, mentoring Grace: Grace's group lists its own
    # mentors, which the add below does not change, and neither does it
    # change Barbara's group.
    barbara = Person.objects.create(name="Barbara")
    barbara.mentees.add(grace)

    # Added from the mentor's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with Grace as its instance and Ada's primary
    # key in pk_set, so the person to replace is the one named in pk_set, the
    # mentee, not the instance. No row is saved again: the add writes only
    # Ada's link to its new mentor.
    with django_capture_on_commit_callbacks(execute=True):
        grace.mentees.add(ada)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Ada's group as committed, with Grace's name added after its other
    # mentor's, and no group of the person not mentoring Ada. Grace's own
    # group is not replaced: it follows its mentors, not its mentees, so the
    # add does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nLinus\n\nGrace",
                source_app_label="testapp",
                source_model="person",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
    }


@pytest.mark.django_db
def test_removing_a_mentee_from_a_mentor_replaces_the_mentors_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the remove below is observed.
    # Ada has two mentors: Grace, which mentors another person too, so that its
    # group keeps the mentee left, and Linus, which Ada keeps.
    ada = Person.objects.create(name="Ada")
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    linus = Person.objects.create(name="Linus")
    grace.mentees.add(ada, alan)
    linus.mentees.add(ada)

    # Removed from the mentor's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with Grace as its instance and Ada's primary
    # key in pk_set, so the person to replace is the instance, the mentor,
    # not the one named in pk_set. No row is saved again: the remove deletes
    # only Grace's link to Ada.
    with django_capture_on_commit_callbacks(execute=True):
        grace.mentees.remove(ada)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Grace's group as committed, without Ada's name, its other mentee kept,
    # and no group of the mentor Ada keeps. Ada's own group is not replaced:
    # it follows its mentees, not its mentors, so the remove does not change
    # it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace\n\nAlan",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_mentees_of_a_mentor_replaces_only_the_mentors_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Grace has two mentees: Alan mentors a person of its own, so that a
    # replacement of its group would show that mentee, Ada mentors no one.
    ada = Person.objects.create(name="Ada")
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    linus = Person.objects.create(name="Linus")
    grace.mentees.add(ada, alan)
    alan.mentees.add(linus)
    # A person neither mentored by Grace nor mentoring Grace: the clear below
    # does not change Barbara's group.
    barbara = Person.objects.create(name="Barbara")
    barbara.mentees.add(linus)

    # Cleared from the mentor's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with Grace as its instance and no primary keys
    # at all, so the person to replace is the instance, the mentor, not its
    # former mentees. No row is saved again: the clear deletes only Grace's
    # links to its mentees.
    with django_capture_on_commit_callbacks(execute=True):
        grace.mentees.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Grace's group as committed, without the names of its former mentees,
    # and no group of the person unrelated to the clear. The groups of the
    # former mentees are not replaced: each follows its own mentees, not its
    # mentors, so the clear does not change them.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_mentors_of_a_person_replaces_the_group_of_each_former_mentor(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed. Ada
    # has two mentors: one mentors another person too, so that its group keeps
    # the mentee left, the other mentors it alone.
    ada = Person.objects.create(name="Ada")
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    linus = Person.objects.create(name="Linus")
    ada.mentors.add(grace, linus)
    alan.mentors.add(grace)
    # A person not mentoring Ada: the clear below does not change Barbara's
    # group.
    barbara = Person.objects.create(name="Barbara")
    alan.mentors.add(barbara)

    # Cleared from the mentee's side: Django sends m2m_changed with Ada as its
    # instance and no primary keys at all, so its mentors can only be found
    # before the clear, by the join rows naming it as the mentee, not as the
    # mentor. No row is saved again: the clear deletes only Ada's links to its
    # mentors.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of both former mentors as committed, each without Ada's
    # name, and no group of the person that never mentored it. Ada's own
    # group is not replaced: it follows its mentees, not its mentors, so the
    # clear does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{grace.pk}": [
            NormalizedDocument(
                text="Grace\n\nAlan",
                source_app_label="testapp",
                source_model="person",
                source_pk=grace.pk,
                title="Grace",
            ),
        ],
        f"testapp.person:{linus.pk}": [
            NormalizedDocument(
                text="Linus",
                source_app_label="testapp",
                source_model="person",
                source_pk=linus.pk,
                title="Linus",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_mentees_of_a_mentor_following_mentors_replaces_each_former_mentee(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_persons_following_their_mentors()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves and these adds never run, so only the clear below is observed.
    # Grace has two mentees: Alan has another mentor too, so that its group
    # keeps the mentor left, Ada has Grace alone.
    ada = Person.objects.create(name="Ada")
    alan = Person.objects.create(name="Alan")
    grace = Person.objects.create(name="Grace")
    linus = Person.objects.create(name="Linus")
    grace.mentees.add(ada, alan)
    linus.mentees.add(alan)
    # Grace has a mentor of her own: her group lists it, and the clear below
    # does not change it.
    barbara = Person.objects.create(name="Barbara")
    barbara.mentees.add(grace)
    # A person Grace never mentored: the clear below does not change
    # Margaret's group.
    margaret = Person.objects.create(name="Margaret")
    linus.mentees.add(margaret)

    # Cleared from the mentor's side, the reverse side of the many-to-many:
    # Django sends m2m_changed with Grace as its instance and no primary keys
    # at all, so its mentees can only be found before the clear, by the join
    # rows naming it as the mentor, not as the mentee. No row is saved again:
    # the clear deletes only Grace's links to its mentees.
    with django_capture_on_commit_callbacks(execute=True):
        grace.mentees.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The groups of both former mentees as committed, each without Grace's
    # name, and no group of the person Grace never mentored. Grace's own group
    # is not replaced: it follows its mentors, not its mentees, so the clear
    # does not change it.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{ada.pk}": [
            NormalizedDocument(
                text="Ada",
                source_app_label="testapp",
                source_model="person",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
        f"testapp.person:{alan.pk}": [
            NormalizedDocument(
                text="Alan\n\nLinus",
                source_app_label="testapp",
                source_model="person",
                source_pk=alan.pk,
                title="Alan",
            ),
        ],
    }


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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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
    carving = _create_the_carving_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking, carving)

    # tests/settings.py defines no MODEL_RAG_OUTPUT.

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

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

    _register_courses_following_their_topics()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the topic's delete below is observed. The
    # course covers two topics, so that its group keeps the one left.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
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

    _register_topics_following_their_courses()

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
    hall = _create_the_halle_tony_garnier()
    # Another venue of the same city, with a seminar of its own: only both
    # columns together name a venue, so a seminar at the hall is not one of
    # its seminars.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        _create_a_seminar_at(hall)
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
def test_updating_a_seminar_of_a_venue_following_by_multi_column_reads_it_at_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_venues_following_their_seminars()

    # Created outside the counted queries: only the seminar's update below is
    # observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()

    # The queries are counted around the commit callbacks too, which run when
    # the inner context exits: the lookup of the Venue before the save, the
    # save's UPDATE, the lookup of the Venue from the row as committed, then
    # the reads of the Venue's group. The Venue is not also read at post_save
    # by the two columns the seminar holds in memory: the lookup at the commit
    # already finds it.
    with (
        django_assert_num_queries(6) as queries,
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Saved again in place: the seminar's two columns still name the same
        # venue, neither of them its primary key.
        acoustics.title = "Acoustics of large halls"
        acoustics.save()

    # No query turns the two columns the seminar holds into the Venue's
    # primary key.
    by_columns = 'FROM "testapp_venue" WHERE ("testapp_venue"."city"'
    assert not [
        query for query in queries.captured_queries if by_columns in query["sql"]
    ]
    # The Venue's group, once, with the seminar's text fields as committed
    # after the Venue's own name.
    assert _replaced(built_outputs) == [
        {
            f"testapp.venue:{hall.pk}": [
                NormalizedDocument(
                    text=(
                        "Halle Tony Garnier\n\nAcoustics of large halls"
                        "\n\nLyon\n\nHalle Tony Garnier"
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
    _register_venues_by_their_seminar_titles()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the seminar's save below is observed.
    hall = _create_the_halle_tony_garnier()
    # Another venue of the same city, with a seminar of its own: only both
    # columns together name a venue, so a seminar at the hall is not one of
    # its seminars.
    _create_the_transbordeur_and_its_seminar()

    with django_capture_on_commit_callbacks(execute=True):
        _create_a_seminar_at(hall)
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
def test_deleting_a_venue_read_through_a_multi_column_lookup_path_sends_only_seminars(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, reading its venue's name through a lookup
    # path across the multi-column ForeignObject ``venue``, CASCADE on delete:
    # Venue itself is not.
    _register_seminars_reading_their_venue_through_a_path()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's delete below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    rigging = _create_a_seminar_at(hall, title="Rigging")
    acoustics_pk = acoustics.pk
    rigging_pk = rigging.pk
    # Another venue of the same city: only both columns together name a venue,
    # so its seminar is neither deleted with the hall nor sent.
    _create_the_transbordeur_and_its_seminar()

    # The seminars are deleted with the venue by cascade: they are found as its
    # readers before the delete, yet no longer exist at the commit.
    with django_capture_on_commit_callbacks(execute=True):
        hall.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The seminars' empty groups, merged across replace calls: whether they
    # come in one call or one per seminar is not what this test is about. No
    # group still carrying the deleted venue's name, none for the venue.
    assert _received_groups(built_outputs) == {
        f"testapp.seminar:{acoustics_pk}": [],
        f"testapp.seminar:{rigging_pk}": [],
    }
    # Each group sent once: no replacement of a seminar's group follows its
    # empty one.
    sent_source_keys = [
        source_key for groups in _replaced(built_outputs) for source_key in groups
    ]
    assert sorted(sent_source_keys) == sorted(
        [f"testapp.seminar:{acoustics_pk}", f"testapp.seminar:{rigging_pk}"]
    )


@pytest.mark.django_db
def test_saving_a_venue_unchanged_replaces_each_of_its_following_seminars_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Seminar is registered, following its venue through the
    # multi-column ForeignObject ``venue``: Venue itself is not.
    _register_seminars_following_their_venue()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the venue's save below is observed.
    hall, acoustics = _create_the_hall_and_its_acoustics_seminar()
    rigging = _create_a_seminar_at(hall, title="Rigging")

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
        lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
        lamps = Category.objects.create(name="Lamps")
        tools = _create_the_tools_category()
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
        lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
        lamp = _create_a_clearance_desk_lamp(lighting)
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
def test_deleting_a_page_whose_followed_plugins_cascade_sends_only_its_empty_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_pages_following_their_plugins()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the Page's delete below is observed.
    page = _create_the_about_page()
    page_pk = page.pk
    _create_a_plugin_on(page)
    _create_a_plugin_on(page, "We ship worldwide.")

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

    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)

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
    _register_workshops_following_their_topic()
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
    page = _create_the_about_page()

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
    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)

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

    lighting = _create_the_lighting_category()

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

    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)

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

    page = _create_the_about_page()
    plugin = _create_a_plugin_on(page)
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


@pytest.mark.django_db(transaction=True)
def test_a_plugin_saved_in_a_rolled_back_transaction_page_following_it_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Created before Page is registered: in autocommit (transaction=True),
    # their own commit callbacks would otherwise run, and send the page's group.
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)

    _register_pages_following_their_plugins()

    # A real transaction (transaction=True), so that leaving the atomic block
    # on an exception rolls it back rather than a savepoint of the test's own.
    with pytest.raises(_RolledBackError), transaction.atomic():
        chairs.body = "We build chairs and tables by hand."
        chairs.save()
        raise _RolledBackError

    # Not the page's group, nor any other: no replace call at all.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db(transaction=True)
def test_a_topic_added_to_a_course_in_a_rolled_back_transaction_sends_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Created before Course is registered, and not linked: in autocommit
    # (transaction=True), their own commit callbacks would otherwise run, and
    # send the course's group.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")

    _register_courses_following_their_topics()

    # A real transaction (transaction=True), so that leaving the atomic block
    # on an exception rolls it back rather than a savepoint of the test's own.
    with pytest.raises(_RolledBackError), transaction.atomic():
        basics.topics.add(woodworking)
        raise _RolledBackError

    # Not the course's group, nor any other: no replace call at all.
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
    page = _create_the_about_page()

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
    lighting = _create_the_lighting_category()
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
        lighting = _create_the_lighting_category()
        tools = _create_the_tools_category()
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
        lighting = _create_the_lighting_category()
        tools = _create_the_tools_category()
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
        lighting = _create_the_lighting_category()
        tools = _create_the_tools_category()
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
    _register_products_following_their_category()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the categories' saves below are observed.
    lighting = _create_the_lighting_category()
    desk_lamp = _create_a_plain_desk_lamp(lighting)
    tools = _create_the_tools_category()
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
