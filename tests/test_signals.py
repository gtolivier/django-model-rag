import inspect
import logging
from collections import Counter
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
    BonusTrack,
    Category,
    CategoryProxy,
    ClearanceProduct,
    Course,
    Craftsman,
    Exhibit,
    FeaturedProduct,
    Friend,
    Guild,
    Lesson,
    Listing,
    Match,
    Membership,
    Musician,
    Offer,
    Page,
    Person,
    Photo,
    Product,
    Review,
    Seminar,
    Shelf,
    Showroom,
    Supplier,
    SupplierOrder,
    SupplierProfile,
    Tag,
    Team,
    TextPlugin,
    TextPluginProxy,
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


def _replaced_source_keys(built_outputs: list[TrackedRecordingOutput]) -> list[str]:
    """The source key of every group replaced, across every output built, once
    per replace call that carried it, in call order."""
    return [source_key for groups in _replaced(built_outputs) for source_key in groups]


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


def _lists_every_row_of(model: type[Model], sql: str) -> bool:
    """Whether ``sql`` lists every row of ``model``: a select from its table
    with no WHERE clause, as a lookup of given rows or a cascade would have."""
    table = connection.ops.quote_name(model._meta.db_table)
    return sql.startswith("SELECT") and f"FROM {table}" in sql and "WHERE" not in sql


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


def _register_persons_following_their_mentors_and_mentees() -> None:
    """Register Person once, by its name, following both sides of its own
    many-to-many ``mentors``: its mentors, the forward side, and its mentees,
    the reverse side."""
    rag.register(Person, fields=["name"], follow=["mentors", "mentees"])


def _register_friends_following_their_friends() -> None:
    """Register Friend, by its name, following its friends through its own
    symmetrical many-to-many ``friends``: both sides of the links are Friends,
    and each link is a link back."""
    rag.register(Friend, fields=["name"], follow=["friends"])


def _register_teams_following_their_matches() -> None:
    """Register Team, by its name, following its matches by both reverse foreign
    keys, ``home_matches`` and ``away_matches``: Match itself is not."""
    rag.register(Team, fields=["name"], follow=["home_matches", "away_matches"])


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


def _create_the_whittling_course_on_carving() -> Course:
    """Create the Carving topic, then the Whittling basics course, linked to it."""
    carving = _create_the_carving_topic()
    whittling = Course.objects.create(title="Whittling basics")
    whittling.topics.add(carving)
    return whittling


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


def _create_the_news_page() -> Page:
    """Create the News page."""
    return Page.objects.create(title="News", slug="news")


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


def _create_the_lighting_category() -> Category:
    """Create the Lighting category."""
    return Category.objects.create(name="Lighting")


def _create_the_tools_category() -> Category:
    """Create the Tools category."""
    return Category.objects.create(name="Tools")


def _register_products_following_their_category() -> None:
    """Register only Product, by its name, following its category through its
    own foreign key: Category itself is not registered."""
    rag.register(Product, fields=["name"], follow=["category"])


def _register_workshops_following_their_topic() -> None:
    """Register Workshop, following its topic through its own nullable foreign
    key: Topic itself is not registered."""
    rag.register(Workshop, follow=["topic"])


def _create_a_pottery_workshop(topic: Topic) -> Workshop:
    """Create a Pottery workshop on the given topic."""
    return Workshop.objects.create(title="Pottery", topic=topic)


def _register_products_reading_their_category_through_a_path() -> None:
    """Register Product, reading its category's name through a lookup path,
    with no follow: Category itself is not registered."""
    rag.register(Product, fields=["name", "category__name"])


def _register_offers_reading_their_category_two_foreign_keys_deep() -> None:
    """Register Offer, reading its product's category's name through a lookup
    path two foreign keys deep, with no follow: neither Product nor Category is
    registered."""
    rag.register(Offer, fields=["title", "product__category__name"])


@dataclass(frozen=True, kw_only=True)
class _OfferInEachCategory:
    """The rows _create_an_offer_in_tools_and_in_lighting() creates that the
    tests name."""

    tools: Category
    hammer_deal: Offer
    lighting: Category
    lamp: Product
    lamp_sale: Offer


def _create_an_offer_in_tools_and_in_lighting() -> _OfferInEachCategory:
    """Create the Tools category, a Hammer in it and its Hammer deal offer, then
    the Lighting category, a Desk lamp in it and its Lamp sale offer."""
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)
    hammer_deal = Offer.objects.create(title="Hammer deal", product=hammer)
    lighting = _create_the_lighting_category()
    lamp = _create_a_plain_desk_lamp(lighting)
    lamp_sale = Offer.objects.create(title="Lamp sale", product=lamp)
    return _OfferInEachCategory(
        tools=tools,
        hammer_deal=hammer_deal,
        lighting=lighting,
        lamp=lamp,
        lamp_sale=lamp_sale,
    )


def _create_the_garden_category() -> Category:
    """Create the Garden category."""
    return Category.objects.create(name="Garden")


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
def test_a_product_in_the_middle_of_an_offer_lookup_path_saved_replaces_every_offer(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: neither
    # Product nor Category is registered, so the product is a link in the
    # middle of the path.
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the product's save below is observed. One offer is on the
    # product saved, the other on another product, in another category.
    offers = _create_an_offer_in_tools_and_in_lighting()
    hammer_deal, lamp, lamp_sale = offers.hammer_deal, offers.lamp, offers.lamp_sale

    with django_capture_on_commit_callbacks(execute=True):
        lamp.name = "Reading lamp"
        lamp.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every offer is replaced, as committed, the one on
    # the other product included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{hammer_deal.pk}": [
            NormalizedDocument(
                text="Hammer deal\n\nTools",
                source_app_label="testapp",
                source_model="offer",
                source_pk=hammer_deal.pk,
                title="Hammer deal",
            ),
        ],
        f"testapp.offer:{lamp_sale.pk}": [
            NormalizedDocument(
                text="Lamp sale\n\nLighting",
                source_app_label="testapp",
                source_model="offer",
                source_pk=lamp_sale.pk,
                title="Lamp sale",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_page_a_plugin_extractor_depends_on_saved_replaces_every_plugin(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the TextPlugin is registered, with a custom extractor whose
    # depends_on names its page: Page itself is not registered.
    _register_plugins_by_their_page_title()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the page's save below is observed. One plugin is on the
    # page saved, the other on another page.
    workshop = _create_the_workshop_page()
    shipping = _create_a_plugin_on(workshop, body=_SIBLING_PLUGIN_BODY)
    about = _create_the_about_page()
    chairs = _create_a_plugin_on(about)

    with django_capture_on_commit_callbacks(execute=True):
        about.title = "Who we are"
        about.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every plugin is replaced, as committed, the one on
    # the other page included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.textplugin:{shipping.pk}": [
            NormalizedDocument(
                text="Our workshop: We ship worldwide.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=shipping.pk,
            ),
        ],
        f"testapp.textplugin:{chairs.pk}": [
            NormalizedDocument(
                text="Who we are: We build chairs by hand.",
                source_app_label="testapp",
                source_model="textplugin",
                source_pk=chairs.pk,
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
            [TextPlugin(page=about, body=_PLUGIN_BODY)]
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
    news = _create_the_news_page()
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
    news = _create_the_news_page()

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
        chairs = _create_a_plugin_on(about)
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
def test_a_category_followed_by_foreign_key_saved_replaces_every_product_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    _register_products_following_their_category()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the category's save below is observed. One product is in
    # the category saved, the other in another category.
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)
    lighting = _create_the_lighting_category()
    lamp = _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every product is replaced, as committed, the one
    # in the other category included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{hammer.pk}": [
            NormalizedDocument(
                text="Hammer\n\nTools",
                source_app_label="testapp",
                source_model="product",
                source_pk=hammer.pk,
                title="Hammer",
                url=f"/products/{hammer.pk}/",
            ),
        ],
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
def test_a_category_created_then_bulk_given_a_product_resyncs_every_product_at_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: Category itself is not.
    _register_products_following_their_category()

    # Created outside the captured callbacks: their commit callbacks never
    # run. The product is in another category than the one created below.
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)

    with django_capture_on_commit_callbacks(execute=True):
        lighting = _create_the_lighting_category()
        # Attached to the category after its creation, in the same
        # transaction, by a write that sends no signal: only the category's
        # creation is observed.
        (lamp,) = Product.objects.bulk_create(
            [
                Product(
                    name="Desk lamp",
                    description="A lamp for the desk.",
                    price="25.00",
                    category=lighting,
                )
            ]
        )

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The creation resyncs the whole registered model at the
    # commit: the group of every product is replaced, as committed, the one
    # attached after the creation included, and the one in the other category,
    # unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{hammer.pk}": [
            NormalizedDocument(
                text="Hammer\n\nTools",
                source_app_label="testapp",
                source_model="product",
                source_pk=hammer.pk,
                title="Hammer",
                url=f"/products/{hammer.pk}/",
            ),
        ],
        f"testapp.product:{lamp.pk}": [
            NormalizedDocument(
                text="Desk lamp\n\nLighting",
                source_app_label="testapp",
                source_model="product",
                source_pk=lamp.pk,
                title="Desk lamp",
                url=f"/products/{lamp.pk}/",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_profile_read_past_a_reverse_one_to_one_saved_replaces_every_supplier_order(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the SupplierOrder is registered, reading its supplier's profile body
    # through a lookup path that crosses the foreign key `supplier`, then the
    # reverse one-to-one by its query name (supplier_profile), which differs
    # from its accessor (profile). Neither Supplier nor SupplierProfile is
    # registered, so the profile reaches the orders only through the path.
    rag.register(
        SupplierOrder, fields=["reference", "supplier__supplier_profile__body"]
    )

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the profile's save below is observed. One order is from
    # the supplier whose profile is saved, the other from another supplier,
    # with a profile of its own.
    birch_mill = _create_birch_mill()
    birch_profile = _create_a_profile_of(birch_mill, _PROFILE_BODY)
    birch_order = SupplierOrder.objects.create(reference="PO-1", supplier=birch_mill)
    oak_yard = _create_oak_yard()
    _create_a_profile_of(oak_yard, "Seasoned oak beams.")
    oak_order = SupplierOrder.objects.create(reference="PO-2", supplier=oak_yard)

    with django_capture_on_commit_callbacks(execute=True):
        birch_profile.body = "Air-dried planks."
        birch_profile.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every order is replaced, as committed, the one
    # from the other supplier included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.supplierorder:{birch_order.pk}": [
            NormalizedDocument(
                text="PO-1\n\nAir-dried planks.",
                source_app_label="testapp",
                source_model="supplierorder",
                source_pk=birch_order.pk,
                title="PO-1",
            ),
        ],
        f"testapp.supplierorder:{oak_order.pk}": [
            NormalizedDocument(
                text="PO-2\n\nSeasoned oak beams.",
                source_app_label="testapp",
                source_model="supplierorder",
                source_pk=oak_order.pk,
                title="PO-2",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_category_saved_through_a_proxy_replaces_every_product_following_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, following its category through its own
    # foreign key: neither Category nor its proxy is.
    _register_products_following_their_category()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the proxy's save below is observed. One product is in the
    # category saved, the other in another category.
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)
    lighting = _create_the_lighting_category()
    lamp = _create_a_plain_desk_lamp(lighting)

    # Django sends post_save with the proxy as its sender, not Category.
    with django_capture_on_commit_callbacks(execute=True):
        lighting_proxy = CategoryProxy.objects.get(pk=lighting.pk)
        lighting_proxy.name = "Lamps"
        lighting_proxy.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every product is replaced, as committed, the one
    # in the other category included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{hammer.pk}": [
            NormalizedDocument(
                text="Hammer\n\nTools",
                source_app_label="testapp",
                source_model="product",
                source_pk=hammer.pk,
                title="Hammer",
                url=f"/products/{hammer.pk}/",
            ),
        ],
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
def test_a_topic_followed_by_a_set_null_foreign_key_deleted_replaces_every_workshop(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Workshop is registered, following its topic through its own
    # foreign key, SET_NULL on delete: Topic itself is not.
    _register_workshops_following_their_topic()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the topic's delete below is observed. One workshop is on
    # the topic deleted, the other on another topic.
    woodworking = _create_the_woodworking_topic()
    pottery = _create_a_pottery_workshop(woodworking)
    carving = _create_the_carving_topic()
    whittling = Workshop.objects.create(title="Whittling", topic=carving)

    with django_capture_on_commit_callbacks(execute=True):
        woodworking.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The delete resyncs the whole registered model at the
    # commit: the group of every workshop is replaced, as committed, the one
    # on the other topic included, unchanged; the one on the deleted topic
    # has lost it, its foreign key set to null.
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
        f"testapp.workshop:{whittling.pk}": [
            NormalizedDocument(
                text="Whittling\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="workshop",
                source_pk=whittling.pk,
                title="Whittling",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_topic_followed_by_many_to_many_saved_replaces_every_course_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many: Topic itself is not.
    _register_courses_following_their_topics()

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the topic's save below is observed. One
    # course has the topic saved, the other only another topic.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)
    whittling = _create_the_whittling_course_on_carving()

    with django_capture_on_commit_callbacks(execute=True):
        # A save alone: no link is written, so no m2m_changed is sent.
        woodworking.title = "Joinery"
        woodworking.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every course is replaced, as committed, the one
    # without the topic saved included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nJoinery\n\nJoints and finishes.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
        f"testapp.course:{whittling.pk}": [
            NormalizedDocument(
                text="Whittling basics\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="course",
                source_pk=whittling.pk,
                title="Whittling basics",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_course_followed_by_reverse_many_to_many_saved_replaces_every_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Topic is registered, following its courses by the reverse side
    # of Course's many-to-many: Course itself is not.
    _register_topics_following_their_courses()

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the course's save below is observed. One
    # topic is in the course saved, the other only in another course.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    basics.topics.add(woodworking)
    carving = _create_the_carving_topic()
    whittling = Course.objects.create(title="Whittling basics")
    whittling.topics.add(carving)

    with django_capture_on_commit_callbacks(execute=True):
        # A save alone: no link is written, so no m2m_changed is sent.
        basics.title = "Joinery basics"
        basics.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every topic is replaced, as committed, the one
    # outside the course saved included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.topic:{woodworking.pk}": [
            NormalizedDocument(
                text="Woodworking\n\nJoints and finishes.\n\nJoinery basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=woodworking.pk,
                title="Woodworking",
            ),
        ],
        f"testapp.topic:{carving.pk}": [
            NormalizedDocument(
                text="Carving\n\nKnives and gouges.\n\nWhittling basics",
                source_app_label="testapp",
                source_model="topic",
                source_pk=carving.pk,
                title="Carving",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_tag_followed_by_generic_relation_saved_replaces_every_photo_at_the_commit(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Photo is registered, following its tags through its generic
    # relation: Tag itself is not.
    rag.register(Photo, follow=["tags"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the tag's save below is observed. One photo carries the
    # tag saved, the other only a tag of its own.
    workbench = Photo.objects.create(title="Workbench")
    oak = Tag.objects.create(content_object=workbench, label="Oak")
    lathe = Photo.objects.create(title="Lathe")
    Tag.objects.create(content_object=lathe, label="Turning")

    with django_capture_on_commit_callbacks(execute=True):
        oak.label = "Walnut"
        oak.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every photo is replaced, as committed, the one
    # without the tag saved included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.photo:{workbench.pk}": [
            NormalizedDocument(
                text="Workbench\n\nWalnut",
                source_app_label="testapp",
                source_model="photo",
                source_pk=workbench.pk,
                title="Workbench",
            ),
        ],
        f"testapp.photo:{lathe.pk}": [
            NormalizedDocument(
                text="Lathe\n\nTurning",
                source_app_label="testapp",
                source_model="photo",
                source_pk=lathe.pk,
                title="Lathe",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_shelf_followed_by_a_to_field_foreign_key_saved_replaces_every_warehouse(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Warehouse is registered, following its shelves by the reverse
    # side of Shelf's foreign key, which points to the warehouse's unique
    # code, not its primary key: Shelf itself is not registered.
    rag.register(Warehouse, fields=["name"], follow=["shelves"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the shelf's save below is observed. One warehouse holds the
    # shelf saved, the other only a shelf of its own.
    north = Warehouse.objects.create(name="North depot", code="north")
    timber = Shelf.objects.create(label="Timber", warehouse=north)
    south = Warehouse.objects.create(name="South depot", code="south")
    Shelf.objects.create(label="Glue", warehouse=south)

    with django_capture_on_commit_callbacks(execute=True):
        timber.label = "Plywood"
        timber.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every warehouse is replaced, as committed, the one
    # without the shelf saved included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.warehouse:{north.pk}": [
            NormalizedDocument(
                text="North depot\n\nPlywood",
                source_app_label="testapp",
                source_model="warehouse",
                source_pk=north.pk,
                title="North depot",
            ),
        ],
        f"testapp.warehouse:{south.pk}": [
            NormalizedDocument(
                text="South depot\n\nGlue",
                source_app_label="testapp",
                source_model="warehouse",
                source_pk=south.pk,
                title="South depot",
            ),
        ],
    }


@pytest.mark.django_db
def test_no_related_resync_a_to_field_shelf_saved_replaces_its_own_warehouse_alone(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Warehouse is registered, following its shelves by the reverse
    # side of Shelf's foreign key, which points to the warehouse's unique
    # code, not its primary key, with the resync on related writes turned
    # off: a shelf still replaces its own warehouse's group, and no other.
    # Shelf itself is not registered.
    rag.register(
        Warehouse,
        fields=["name"],
        follow=["shelves"],
        resync_on_related_writes=False,
    )

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the shelf's save below is observed. One warehouse holds the
    # shelf saved, the other only a shelf of its own.
    north = Warehouse.objects.create(name="North depot", code="north")
    timber = Shelf.objects.create(label="Timber", warehouse=north)
    south = Warehouse.objects.create(name="South depot", code="south")
    Shelf.objects.create(label="Glue", warehouse=south)

    with django_capture_on_commit_callbacks(execute=True):
        timber.label = "Plywood"
        timber.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Only the group of the saved shelf's own warehouse is
    # replaced, as committed: not the other warehouse's.
    assert _received_groups(built_outputs) == {
        f"testapp.warehouse:{north.pk}": [
            NormalizedDocument(
                text="North depot\n\nPlywood",
                source_app_label="testapp",
                source_model="warehouse",
                source_pk=north.pk,
                title="North depot",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_seminar_followed_by_a_multi_column_relation_saved_replaces_every_venue(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Venue is registered, following its seminars by the reverse side
    # of Seminar's ForeignObject, which matches the venue by its city and name,
    # two columns, not its primary key: Seminar itself is not registered.
    rag.register(Venue, fields=["name"], follow=["seminars"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the seminar's save below is observed. One venue hosts the
    # seminar saved, the other only a seminar of its own.
    hall = Venue.objects.create(city="Lyon", name="Town hall")
    joinery = Seminar.objects.create(
        title="Joinery", venue_city="Lyon", venue_name="Town hall"
    )
    library = Venue.objects.create(city="Lyon", name="Library")
    Seminar.objects.create(title="Bookbinding", venue_city="Lyon", venue_name="Library")

    with django_capture_on_commit_callbacks(execute=True):
        joinery.title = "Marquetry"
        joinery.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every venue is replaced, as committed, the one
    # without the seminar saved included, unchanged. Each seminar reads as its
    # text columns, the two matching its venue included.
    assert _received_groups(built_outputs) == {
        f"testapp.venue:{hall.pk}": [
            NormalizedDocument(
                text="Town hall\n\nMarquetry\n\nLyon\n\nTown hall",
                source_app_label="testapp",
                source_model="venue",
                source_pk=hall.pk,
                title="Town hall",
            ),
        ],
        f"testapp.venue:{library.pk}": [
            NormalizedDocument(
                text="Library\n\nBookbinding\n\nLyon\n\nLibrary",
                source_app_label="testapp",
                source_model="venue",
                source_pk=library.pk,
                title="Library",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_product_saved_through_a_multi_table_child_replaces_every_review(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Review is registered, following its product through its own
    # foreign key to Product: neither Product nor FeaturedProduct is.
    rag.register(Review, follow=["product"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the featured product's save below is observed. One review
    # is of the product saved, the other of another product.
    lamp = _create_a_desk_lamp(_create_the_lighting_category())
    sturdy = Review.objects.create(title="Sturdy", product=lamp)
    hammer = _create_a_hammer(_create_the_tools_category())
    heavy = Review.objects.create(title="Heavy", product=hammer)

    # Django sends post_save with FeaturedProduct as its sender, not Product,
    # though the save writes the Product row the reviews read.
    with django_capture_on_commit_callbacks(execute=True):
        lamp.name = "Reading lamp"
        lamp.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every review is replaced, as committed, the one of
    # the other product included, unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.review:{sturdy.pk}": [
            NormalizedDocument(
                text="Sturdy\n\nReading lamp\n\nA lamp for the desk.\n\nNew",
                source_app_label="testapp",
                source_model="review",
                source_pk=sturdy.pk,
                title="Sturdy",
            ),
        ],
        f"testapp.review:{heavy.pk}": [
            NormalizedDocument(
                text="Heavy\n\nHammer\n\nA hammer for nails.\n\nNew",
                source_app_label="testapp",
                source_model="review",
                source_pk=heavy.pk,
                title="Heavy",
            ),
        ],
    }


def _register_new_products_by_their_category_name() -> None:
    """Register only Product, with a custom extractor reading its name, then its
    category's name: depends_on names the category, whose saves change its
    documents. Its get_queryset() leaves out the second-hand products. Category
    itself is not registered."""

    @rag.register_extractor(Product, depends_on=["category"])
    class NewProductExtractor(BaseExtractor[Product]):
        def get_queryset(self, queryset: QuerySet[Product]) -> QuerySet[Product]:
            return queryset.filter(condition="new")

        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.name}\n\n{instance.category.name}"
            )


@pytest.mark.django_db
def test_a_category_saved_empties_the_group_of_a_product_get_queryset_leaves_out(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, with a custom extractor whose
    # depends_on names its category and whose get_queryset() keeps only the
    # new products: Category itself is not registered.
    _register_new_products_by_their_category_name()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the category's save below is observed. One product is in
    # the category saved; the two others are in another category, one of them
    # second-hand: only the resync of the whole model reaches it.
    lighting = _create_the_lighting_category()
    lamp = _create_a_plain_desk_lamp(lighting)
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)
    used_saw = Product.objects.create(
        name="Saw",
        description="A saw for planks.",
        price="10.00",
        category=tools,
        condition="used",
    )

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. The save resyncs the whole registered model at the
    # commit: the group of every product it keeps is replaced, as committed,
    # and the second-hand one, which get_queryset() leaves out, gets an empty
    # group, removing whatever the output still holds for it.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{lamp.pk}": [
            NormalizedDocument(
                text="Desk lamp\n\nLamps",
                source_app_label="testapp",
                source_model="product",
                source_pk=lamp.pk,
            ),
        ],
        f"testapp.product:{hammer.pk}": [
            NormalizedDocument(
                text="Hammer\n\nTools",
                source_app_label="testapp",
                source_model="product",
                source_pk=hammer.pk,
            ),
        ],
        f"testapp.product:{used_saw.pk}": [],
    }


@pytest.mark.django_db
def test_several_writes_to_models_an_offer_reads_in_one_transaction_resync_it_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: neither
    # Product nor Category is registered, and the Offer reads both.
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the writes below are observed. One offer is on the product
    # and in the category saved, the other on another product, in another
    # category.
    offers = _create_an_offer_in_tools_and_in_lighting()
    hammer_deal, lamp_sale = offers.hammer_deal, offers.lamp_sale
    lighting, lamp = offers.lighting, offers.lamp

    with django_capture_on_commit_callbacks(execute=True):
        # Two saves of one model the Offer reads, then one of another.
        lamp.name = "Reading lamp"
        lamp.save()
        lamp.price = "30.00"
        lamp.save()
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # However many writes touch the models the Offer reads, the commit
    # resyncs it once: each offer's group is replaced in exactly one replace
    # call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.offer:{hammer_deal.pk}": 1,
        f"testapp.offer:{lamp_sale.pk}": 1,
    }
    # Each replaced as committed, after every write of the transaction.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{hammer_deal.pk}": [
            NormalizedDocument(
                text="Hammer deal\n\nTools",
                source_app_label="testapp",
                source_model="offer",
                source_pk=hammer_deal.pk,
                title="Hammer deal",
            ),
        ],
        f"testapp.offer:{lamp_sale.pk}": [
            NormalizedDocument(
                text="Lamp sale\n\nLamps",
                source_app_label="testapp",
                source_model="offer",
                source_pk=lamp_sale.pk,
                title="Lamp sale",
            ),
        ],
    }


@pytest.mark.django_db
def test_several_writes_to_categories_list_the_products_once_at_the_commit(
    settings: Settings,
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Product is registered, reading its category's name through a
    # lookup path, with no follow: Category itself is not registered, so each
    # write to it resyncs every product at the commit.
    _register_products_reading_their_category_through_a_path()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the writes below are observed. The garden category has no
    # product, so deleting it cascades to none.
    tools = _create_the_tools_category()
    lighting = _create_the_lighting_category()
    garden = _create_the_garden_category()
    _create_a_hammer(tools)
    _create_a_plain_desk_lamp(lighting)

    # The queries captured around the callbacks, so those they run at the
    # commit are captured too.
    with (
        CaptureQueriesContext(connection) as queries,
        django_capture_on_commit_callbacks(execute=True),
    ):
        # Saves and a delete of the model the Product reads, mixed.
        tools.name = "Hand tools"
        tools.save()
        garden.delete()
        lighting.name = "Lamps"
        lighting.save()
        queries_in_the_body = len(queries.captured_queries)

    # However many writes, saves and deletes alike, the products' rows are
    # listed by one query, run at the commit: none while the body runs, a
    # delete included, and not one per write at the commit.
    listing_positions = [
        position
        for position, query in enumerate(queries.captured_queries)
        if _lists_every_row_of(Product, query["sql"])
    ]
    listings_in_the_body = [
        position for position in listing_positions if position < queries_in_the_body
    ]
    listings_at_the_commit = [
        position for position in listing_positions if position >= queries_in_the_body
    ]
    assert (len(listings_in_the_body), len(listings_at_the_commit)) == (0, 1)


@pytest.mark.django_db
def test_a_category_read_by_two_registered_models_saved_resyncs_each_of_them_once(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Two registered models read the Category, neither following the other:
    # the Offer through a lookup path two foreign keys deep, the Listing
    # through a lookup path over its own foreign key, both with no follow.
    # Neither Category nor Product is registered.
    _register_offers_reading_their_category_two_foreign_keys_deep()
    rag.register(Listing, fields=["title", "category__name"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the category's save below is observed. Of each registered
    # model, one row is in the category saved, the other in another category.
    offers = _create_an_offer_in_tools_and_in_lighting()
    hammer_deal, lamp_sale = offers.hammer_deal, offers.lamp_sale
    tools, lighting = offers.tools, offers.lighting
    hammer_rack = Listing.objects.create(
        title="Hammer rack", link="/hammer-rack/", category=tools
    )
    lamp_stand = Listing.objects.create(
        title="Lamp stand", link="/lamp-stand/", category=lighting
    )

    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The one save resyncs both registered models at the commit, each once:
    # every row's group is replaced in exactly one replace call, however the
    # groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.offer:{hammer_deal.pk}": 1,
        f"testapp.offer:{lamp_sale.pk}": 1,
        f"testapp.listing:{hammer_rack.pk}": 1,
        f"testapp.listing:{lamp_stand.pk}": 1,
    }
    # Each replaced as committed, the rows in the other category included,
    # unchanged.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{hammer_deal.pk}": [
            NormalizedDocument(
                text="Hammer deal\n\nTools",
                source_app_label="testapp",
                source_model="offer",
                source_pk=hammer_deal.pk,
                title="Hammer deal",
            ),
        ],
        f"testapp.offer:{lamp_sale.pk}": [
            NormalizedDocument(
                text="Lamp sale\n\nLamps",
                source_app_label="testapp",
                source_model="offer",
                source_pk=lamp_sale.pk,
                title="Lamp sale",
            ),
        ],
        f"testapp.listing:{hammer_rack.pk}": [
            NormalizedDocument(
                text="Hammer rack\n\nTools",
                source_app_label="testapp",
                source_model="listing",
                source_pk=hammer_rack.pk,
                title="Hammer rack",
            ),
        ],
        f"testapp.listing:{lamp_stand.pk}": [
            NormalizedDocument(
                text="Lamp stand\n\nLamps",
                source_app_label="testapp",
                source_model="listing",
                source_pk=lamp_stand.pk,
                title="Lamp stand",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_model_registered_with_no_related_resync_still_replaces_its_own_saved_rows(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow, and with the
    # resync on related writes turned off: a save of a category resyncs no
    # offer. Neither Product nor Category is registered.
    rag.register(
        Offer,
        fields=["title", "product__category__name"],
        resync_on_related_writes=False,
    )

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the writes below are observed. One offer is in each
    # category.
    offers = _create_an_offer_in_tools_and_in_lighting()
    hammer_deal, lighting = offers.hammer_deal, offers.lighting

    with django_capture_on_commit_callbacks(execute=True):
        # A write to the model the Offer reads: the offer in that category
        # reads the new name, yet is not resynced.
        lighting.name = "Lamps"
        lighting.save()
        # A save of an offer itself, in the other category.
        hammer_deal.title = "Hammer bargain"
        hammer_deal.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Only the saved offer's group is replaced, as committed:
    # not the one in the renamed category.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{hammer_deal.pk}": [
            NormalizedDocument(
                text="Hammer bargain\n\nTools",
                source_app_label="testapp",
                source_model="offer",
                source_pk=hammer_deal.pk,
                title="Hammer bargain",
            ),
        ],
    }


def test_no_related_resync_connects_no_delete_listener_to_a_paths_models() -> None:
    # Category and its proxy, and Product: the models the Offer reads only
    # through its lookup path. Neither is registered, nor followed by anything.
    path_models = (Product, Category, CategoryProxy)

    try:
        # Only the Offer is registered, reading its product's category's name
        # through a lookup path two foreign keys deep, with the resync on
        # related writes turned off: no delete of a product or a category
        # resyncs an offer.
        rag.register(
            Offer,
            fields=["title", "product__category__name"],
            resync_on_related_writes=False,
        )

        # Django's deletion Collector fast-deletes only a model with no
        # pre_delete and no post_delete listener: nothing listens to the
        # deletes of the models read only through the path.
        for sender in path_models:
            assert not pre_delete.has_listeners(sender)
            assert not post_delete.has_listeners(sender)
    finally:
        # Undo the registration, then drop any delete listener it left on the
        # path's models, so that none outlives this test and keeps a later one
        # from finding them fast-deleted.
        if Offer in rag.registered_models():
            rag.unregister(Offer)
        for sender in path_models:
            uid = f"django_model_rag.sync_delete.{sender._meta.label}"
            pre_delete.disconnect(sender=sender, dispatch_uid=uid)
            post_delete.disconnect(sender=sender, dispatch_uid=uid)


def test_no_related_resync_extractor_has_no_delete_listener_on_a_fks_model() -> None:
    # Category and its proxy: the models the Product's extractor depends on
    # only through a forward foreign key. Neither is registered, nor followed
    # by anything.
    foreign_key_models = (Category, CategoryProxy)

    try:
        # Only Product is registered, with a custom extractor reading its name
        # and its category's name: depends_on names the category, a forward
        # foreign key, with the resync on related writes turned off: no delete
        # of a category resyncs a product.
        @rag.register_extractor(
            Product, depends_on=["category"], resync_on_related_writes=False
        )
        class CategorizedProductExtractor(BaseExtractor[Product]):
            def extract(self, instance: Product) -> NormalizedDocument:
                return self.build_document(
                    instance, text=f"{instance.name}\n\n{instance.category.name}"
                )

        # Django's deletion Collector fast-deletes only a model with no
        # pre_delete and no post_delete listener: nothing listens to the
        # deletes of the models depended on only through the foreign key.
        for sender in foreign_key_models:
            assert not pre_delete.has_listeners(sender)
            assert not post_delete.has_listeners(sender)
    finally:
        # Undo the registration, then drop any delete listener it left on the
        # foreign key's models, so that none outlives this test and keeps a
        # later one from finding them fast-deleted.
        if Product in rag.registered_models():
            rag.unregister(Product)
        for sender in foreign_key_models:
            uid = f"django_model_rag.sync_delete.{sender._meta.label}"
            pre_delete.disconnect(sender=sender, dispatch_uid=uid)
            post_delete.disconnect(sender=sender, dispatch_uid=uid)


@pytest.mark.django_db
def test_an_extractor_registered_with_no_related_resync_still_replaces_a_childs_parent(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only Product is registered, with a custom extractor reading its name, its
    # category's name, then the titles of its reviews: depends_on names the
    # category, a forward foreign key, and the reviews, a reverse foreign key to
    # its primary key. The resync on related writes is turned off: a save of a
    # category resyncs no product, while a review still replaces its product's
    # group. Neither Category nor Review is registered.
    @rag.register_extractor(
        Product,
        depends_on=["category", "reviews"],
        resync_on_related_writes=False,
    )
    class ReviewedProductExtractor(BaseExtractor[Product]):
        def extract(self, instance: Product) -> NormalizedDocument:
            titles = [review.title for review in instance.reviews.order_by("pk")]
            return self.build_document(
                instance,
                text="\n\n".join([instance.name, instance.category.name, *titles]),
            )

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the writes below are observed. One product is in each
    # category.
    tools = _create_the_tools_category()
    hammer = _create_a_hammer(tools)
    lighting = _create_the_lighting_category()
    _create_a_plain_desk_lamp(lighting)

    with django_capture_on_commit_callbacks(execute=True):
        # A write to a model the extractor reads through a forward foreign key:
        # the product in that category reads the new name, yet is not resynced.
        lighting.name = "Lamps"
        lighting.save()
        # A write to a child row the extractor reads by a reverse foreign key
        # to the product's primary key, on the product in the other category.
        Review.objects.create(title="Heavy", product=hammer)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Merged across replace calls: how the groups are batched is not what this
    # test is about. Only the reviewed product's group is replaced, as
    # committed: not the one in the renamed category.
    assert _received_groups(built_outputs) == {
        f"testapp.product:{hammer.pk}": [
            NormalizedDocument(
                text="Hammer\n\nTools\n\nHeavy",
                source_app_label="testapp",
                source_model="product",
                source_pk=hammer.pk,
            ),
        ],
    }


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
    sent_source_keys = _replaced_source_keys(built_outputs)
    assert sorted(sent_source_keys) == sorted(
        [
            f"testapp.team:{lyon.pk}",
            f"testapp.team:{nantes.pk}",
            f"testapp.tournament:{cup.pk}",
        ]
    )


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
def test_adding_a_topic_to_an_opted_out_course_still_replaces_the_courses_group(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many, with the resync on related writes turned off: a change of
    # its own links still replaces its own group. Topic itself is not
    # registered.
    rag.register(Course, follow=["topics"], resync_on_related_writes=False)

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
def test_adding_a_topic_to_a_course_following_its_topics_resyncs_no_other_course(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many: Topic itself is not.
    _register_courses_following_their_topics()

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the add below is observed. Another course
    # exists, with a topic of its own, untouched by the add.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    _create_the_whittling_course_on_carving()

    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The add is handled on the course's side alone: its group is replaced in
    # exactly one replace call, and no other course is resynced.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {f"testapp.course:{basics.pk}": 1}
    # Replaced as committed: its own title, then the added topic's text.
    assert _received_groups(built_outputs) == {
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


@pytest.mark.django_db
def test_adding_a_course_to_a_topic_courses_following_their_topics_resyncs_every_course(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many: Topic itself is not.
    _register_courses_following_their_topics()

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the add below is observed. Another course
    # exists, with a topic of its own, untouched by the add.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    whittling = _create_the_whittling_course_on_carving()

    # Added from the topic's side: Django sends m2m_changed with the topic,
    # which is not registered, as its instance. Neither row is saved again:
    # the add writes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The instance side is an unregistered topic, so the commit resyncs every
    # course, the untouched one included: each course's group is replaced in
    # exactly one replace call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.course:{basics.pk}": 1,
        f"testapp.course:{whittling.pk}": 1,
    }
    # Each replaced as committed: its own title, then its topic's text.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nWoodworking\n\nJoints and finishes.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
        f"testapp.course:{whittling.pk}": [
            NormalizedDocument(
                text="Whittling basics\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="course",
                source_pk=whittling.pk,
                title="Whittling basics",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_course_to_a_registered_topic_followed_by_courses_resyncs_every_course(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The Course is registered, following its topics through its own
    # many-to-many; Topic is registered too, following nothing, so that its
    # rows' links are read by the courses alone.
    _register_courses_following_their_topics()
    rag.register(Topic)

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the add below is observed. Another course
    # exists, with a topic of its own, untouched by the add.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    whittling = _create_the_whittling_course_on_carving()

    # Added from the topic's side: Django sends m2m_changed with the topic,
    # registered this time, as its instance, and the course among the primary
    # keys it names. Neither row is saved again: the add writes only the link
    # between them.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The courses read the other side of the links, so the commit resyncs
    # every course, the untouched one included, whether or not the topic is
    # registered: each course's group is replaced in exactly one replace call,
    # however the groups are batched. The topic follows nothing, so its own
    # group is not replaced.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.course:{basics.pk}": 1,
        f"testapp.course:{whittling.pk}": 1,
    }
    # Each replaced as committed: its own title, then its topic's text.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nWoodworking\n\nJoints and finishes.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
        f"testapp.course:{whittling.pk}": [
            NormalizedDocument(
                text="Whittling basics\n\nCarving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="course",
                source_pk=whittling.pk,
                title="Whittling basics",
            ),
        ],
    }


@pytest.mark.django_db
def test_clearing_the_courses_of_a_topic_followed_by_courses_resyncs_every_course(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Course is registered, following its topics through its own
    # many-to-many: Topic itself is not.
    _register_courses_following_their_topics()

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the clear below is observed. Two courses
    # cover the topic, so that the clear removes more than one link; another
    # course exists, with a topic of its own, untouched by the clear.
    woodworking = _create_the_woodworking_topic()
    basics = Course.objects.create(title="Woodworking basics")
    joinery = Course.objects.create(title="Joinery")
    woodworking.courses.add(basics, joinery)
    whittling = _create_the_whittling_course_on_carving()

    # Cleared from the topic's side: Django sends m2m_changed with the topic,
    # which is not registered, as its instance. No row is saved again: the
    # clear deletes only the links of the topic.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.clear()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The instance side is an unregistered topic, so the commit resyncs every
    # course, the untouched one included: each course's group is replaced in
    # exactly one replace call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.course:{basics.pk}": 1,
        f"testapp.course:{joinery.pk}": 1,
        f"testapp.course:{whittling.pk}": 1,
    }
    # The cleared courses as committed: the topic's text is gone, only their
    # own title is left.
    received_groups = _received_groups(built_outputs)
    assert received_groups[f"testapp.course:{basics.pk}"] == [
        NormalizedDocument(
            text="Woodworking basics",
            source_app_label="testapp",
            source_model="course",
            source_pk=basics.pk,
            title="Woodworking basics",
        ),
    ]
    assert received_groups[f"testapp.course:{joinery.pk}"] == [
        NormalizedDocument(
            text="Joinery",
            source_app_label="testapp",
            source_model="course",
            source_pk=joinery.pk,
            title="Joinery",
        ),
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
def test_adding_a_course_to_a_topic_resyncs_no_lesson_reading_its_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # The Topic follows its courses; the Lesson reads its topic's title by a
    # lookup path, with no follow. Course itself is not registered.
    _register_topics_following_their_courses()
    rag.register(Lesson, fields=["title", "topic__title"])

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. A lesson exists on
    # the topic the course is added to, another on a topic left untouched.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
    Lesson.objects.create(title="Dovetail joints", topic=woodworking)
    Lesson.objects.create(title="Spoon carving", topic=carving)
    basics = Course.objects.create(title="Woodworking basics")

    # Added from the topic's side: the add writes only the link between the
    # topic and the course. No Topic row and no Course row is saved, so no
    # lesson's text can change.
    with django_capture_on_commit_callbacks(execute=True):
        woodworking.courses.add(basics)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The Topic's group alone is replaced, as committed: its own title and
    # summary, then the added course's title. No lesson's group is replaced.
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
def test_adding_a_topic_to_a_course_both_sides_following_resyncs_every_topic(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Both sides of the many-to-many are registered, each following the other:
    # the Course its topics, the Topic its courses.
    _register_courses_following_their_topics()
    _register_topics_following_their_courses()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Another course and
    # another topic exist, each linked to nothing, untouched by the add.
    woodworking = _create_the_woodworking_topic()
    carving = _create_the_carving_topic()
    basics = Course.objects.create(title="Woodworking basics")
    Course.objects.create(title="Whittling basics")

    # Added from the course's side: Django sends m2m_changed with the course as
    # its instance, and the topic among the primary keys it names. Neither row
    # is saved again: the add writes only the link between them.
    with django_capture_on_commit_callbacks(execute=True):
        basics.topics.add(woodworking)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The course follows its topics, so its group is replaced exactly: the
    # untouched course is not. The topic's courses changed, and Topics read
    # their courses, so the commit resyncs every topic, the untouched one
    # included. Each group is replaced in exactly one replace call, however
    # the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.course:{basics.pk}": 1,
        f"testapp.topic:{woodworking.pk}": 1,
        f"testapp.topic:{carving.pk}": 1,
    }
    # Each replaced as committed: the course's title, then its topic's text;
    # each topic's title and summary, then its course's title, if any.
    assert _received_groups(built_outputs) == {
        f"testapp.course:{basics.pk}": [
            NormalizedDocument(
                text="Woodworking basics\n\nWoodworking\n\nJoints and finishes.",
                source_app_label="testapp",
                source_model="course",
                source_pk=basics.pk,
                title="Woodworking basics",
            ),
        ],
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
                text="Carving\n\nKnives and gouges.",
                source_app_label="testapp",
                source_model="topic",
                source_pk=carving.pk,
                title="Carving",
            ),
        ],
    }


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
def test_creating_a_membership_guilds_following_their_members_resyncs_every_guild(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Guild is registered, following its members through its own
    # many-to-many, whose through model is Membership: neither Craftsman nor
    # Membership is.
    rag.register(Guild, fields=["name"], follow=["members"])

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the create below is observed. Another guild
    # exists, with a member of its own, untouched by the create.
    carpenter = Craftsman.objects.create(name="Carpenter")
    north = Guild.objects.create(name="North guild", code="north")
    smith = Craftsman.objects.create(name="Smith")
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(smith)

    # A row of the through model created directly: Django sends no
    # m2m_changed, only post_save on Membership. Neither the guild nor the
    # craftsman is saved again.
    with django_capture_on_commit_callbacks(execute=True):
        Membership.objects.create(guild=north, craftsman=carpenter)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The guilds read the through model, so the commit resyncs every guild,
    # the untouched one included: each guild's group is replaced in exactly
    # one replace call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.guild:{north.pk}": 1,
        f"testapp.guild:{south.pk}": 1,
    }
    # The changed guild's group as committed: its own name, then its new
    # member's name.
    assert _received_groups(built_outputs)[f"testapp.guild:{north.pk}"] == [
        NormalizedDocument(
            text="North guild\n\nCarpenter",
            source_app_label="testapp",
            source_model="guild",
            source_pk=north.pk,
            title="North guild",
        ),
    ]


@pytest.mark.django_db
def test_deleting_a_membership_guilds_following_their_members_resyncs_every_guild(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Guild is registered, following its members through its own
    # many-to-many, whose through model is Membership: neither Craftsman nor
    # Membership is.
    rag.register(Guild, fields=["name"], follow=["members"])

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the delete below is observed. The north
    # guild has two members, one of which leaves; another guild exists, with a
    # member of its own, untouched by the delete.
    carpenter = Craftsman.objects.create(name="Carpenter")
    mason = Craftsman.objects.create(name="Mason")
    north = Guild.objects.create(name="North guild", code="north")
    leaving = Membership.objects.create(guild=north, craftsman=carpenter)
    Membership.objects.create(guild=north, craftsman=mason)
    smith = Craftsman.objects.create(name="Smith")
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(smith)

    # A row of the through model deleted directly: Django sends no
    # m2m_changed, only post_delete on Membership. Neither the guild nor the
    # craftsman is saved again.
    with django_capture_on_commit_callbacks(execute=True):
        leaving.delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The guilds read the through model, so the commit resyncs every guild,
    # the untouched one included: each guild's group is replaced in exactly
    # one replace call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.guild:{north.pk}": 1,
        f"testapp.guild:{south.pk}": 1,
    }
    # The changed guild's group as committed: its own name, then its remaining
    # member's name only, the removed member gone.
    assert _received_groups(built_outputs)[f"testapp.guild:{north.pk}"] == [
        NormalizedDocument(
            text="North guild\n\nMason",
            source_app_label="testapp",
            source_model="guild",
            source_pk=north.pk,
            title="North guild",
        ),
    ]


@pytest.mark.django_db
def test_deleting_memberships_by_a_queryset_resyncs_every_guild_following_members(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Guild is registered, following its members through its own
    # many-to-many, whose through model is Membership: neither Craftsman nor
    # Membership is.
    rag.register(Guild, fields=["name"], follow=["members"])

    # Created and linked outside the captured callbacks: their commit
    # callbacks never run, so only the delete below is observed. The north
    # guild has three members, two of which leave; another guild exists, with
    # a member of its own, untouched by the delete.
    carpenter = Craftsman.objects.create(name="Carpenter")
    joiner = Craftsman.objects.create(name="Joiner")
    mason = Craftsman.objects.create(name="Mason")
    north = Guild.objects.create(name="North guild", code="north")
    Membership.objects.create(guild=north, craftsman=carpenter)
    Membership.objects.create(guild=north, craftsman=joiner)
    Membership.objects.create(guild=north, craftsman=mason)
    smith = Craftsman.objects.create(name="Smith")
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(smith)

    # Rows of the through model deleted directly, by a queryset: Django sends
    # no m2m_changed. Neither the guilds nor the craftsmen are saved again.
    with django_capture_on_commit_callbacks(execute=True):
        Membership.objects.filter(
            guild=north, craftsman__in=[carpenter, joiner]
        ).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The guilds read the through model, so the commit resyncs every guild,
    # the untouched one included, once for the whole delete: each guild's
    # group is replaced in exactly one replace call, however the groups are
    # batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.guild:{north.pk}": 1,
        f"testapp.guild:{south.pk}": 1,
    }
    # The changed guild's group as committed: its own name, then its remaining
    # member's name only, the removed members gone.
    assert _received_groups(built_outputs)[f"testapp.guild:{north.pk}"] == [
        NormalizedDocument(
            text="North guild\n\nMason",
            source_app_label="testapp",
            source_model="guild",
            source_pk=north.pk,
            title="North guild",
        ),
    ]


@pytest.mark.django_db
def test_deleting_memberships_by_a_queryset_after_a_failed_clear_resyncs_guilds(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # Created and linked before Guild is registered, outside the captured
    # callbacks: with no MODEL_RAG_OUTPUT, their own saves would fail
    # otherwise, and only the delete below is observed. The north guild has
    # three members, two of which leave; another guild exists, with a member
    # of its own, untouched by the delete.
    carpenter = Craftsman.objects.create(name="Carpenter")
    joiner = Craftsman.objects.create(name="Joiner")
    mason = Craftsman.objects.create(name="Mason")
    north = Guild.objects.create(name="North guild", code="north")
    Membership.objects.create(guild=north, craftsman=carpenter)
    Membership.objects.create(guild=north, craftsman=joiner)
    Membership.objects.create(guild=north, craftsman=mason)
    smith = Craftsman.objects.create(name="Smith")
    south = Guild.objects.create(name="South guild", code="south")
    south.members.add(smith)

    # Only the Guild is registered, following its members through its own
    # many-to-many, whose through model is Membership: neither Craftsman nor
    # Membership is.
    rag.register(Guild, fields=["name"], follow=["members"])

    # tests/settings.py defines no MODEL_RAG_OUTPUT: the clear fails before it
    # deletes the memberships, in a block of its own so that the test's
    # transaction goes on.
    with (
        pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"),
        transaction.atomic(),
    ):
        north.members.clear()

    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Rows of the through model deleted directly, by a queryset: Django sends
    # no m2m_changed. Neither the guilds nor the craftsmen are saved again.
    with django_capture_on_commit_callbacks(execute=True):
        Membership.objects.filter(
            guild=north, craftsman__in=[carpenter, joiner]
        ).delete()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The failed clear left nothing behind: the guilds read the through model,
    # so the commit resyncs every guild, the untouched one included, once for
    # the whole delete.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.guild:{north.pk}": 1,
        f"testapp.guild:{south.pk}": 1,
    }
    # The changed guild's group as committed: its own name, then its remaining
    # member's name only, the removed members gone.
    assert _received_groups(built_outputs)[f"testapp.guild:{north.pk}"] == [
        NormalizedDocument(
            text="North guild\n\nMason",
            source_app_label="testapp",
            source_model="guild",
            source_pk=north.pk,
            title="North guild",
        ),
    ]


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
def test_adding_a_mentee_to_a_mentor_replaces_only_the_mentors_group(
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
def test_removing_a_mentee_from_a_mentor_replaces_only_the_mentors_group(
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
def test_adding_a_mentor_to_a_person_following_both_sides_resyncs_every_person(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Person is registered once, following both its mentors and its mentees:
    # each side of its own many-to-many is read by Persons.
    _register_persons_following_their_mentors_and_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Carol is linked to
    # no one, untouched by the add.
    ada = Person.objects.create(name="Ada")
    bob = Person.objects.create(name="Bob")
    carol = Person.objects.create(name="Carol")

    # Added from the mentee's side, the forward side of the many-to-many:
    # Django sends m2m_changed with Ada as its instance and Bob's primary key
    # in pk_set. No row is saved again: the add writes only Ada's link to its
    # new mentor.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.add(bob)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Ada follows her mentors, so her group is replaced; Bob's mentees changed,
    # and Persons read their mentees, so the commit resyncs every person, the
    # untouched one included. Each person's group is replaced in exactly one
    # replace call, however the groups are batched: the resync and the
    # replacement of the instance do not replace Ada's group twice.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.person:{ada.pk}": 1,
        f"testapp.person:{bob.pk}": 1,
        f"testapp.person:{carol.pk}": 1,
    }
    # Each replaced as committed: its own name, then its mentor's or its
    # mentee's, if any.
    assert _received_groups(built_outputs) == {
        f"testapp.person:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nBob",
                source_app_label="testapp",
                source_model="person",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
        f"testapp.person:{bob.pk}": [
            NormalizedDocument(
                text="Bob\n\nAda",
                source_app_label="testapp",
                source_model="person",
                source_pk=bob.pk,
                title="Bob",
            ),
        ],
        f"testapp.person:{carol.pk}": [
            NormalizedDocument(
                text="Carol",
                source_app_label="testapp",
                source_model="person",
                source_pk=carol.pk,
                title="Carol",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_mentor_to_a_person_following_its_mentees_resyncs_every_person(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Person follows only its mentees, the reverse side of its own
    # many-to-many: the forward side, mentors, is read by no one.
    _register_persons_following_their_mentees()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Carol is linked to
    # no one, untouched by the add.
    ada = Person.objects.create(name="Ada")
    bob = Person.objects.create(name="Bob")
    carol = Person.objects.create(name="Carol")

    # Added from the mentee's side, the forward side of the many-to-many:
    # Django sends m2m_changed with Ada as its instance and Bob's primary key
    # in pk_set. No row is saved again: the add writes only Ada's link to its
    # new mentor.
    with django_capture_on_commit_callbacks(execute=True):
        ada.mentors.add(bob)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # Ada's mentors changed, a side no Person reads, so nothing is replaced
    # exactly for her; Bob's mentees changed, and Persons read their mentees,
    # so the commit resyncs every person, Ada and the untouched one included.
    # Each person's group is replaced in exactly one replace call, however the
    # groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.person:{ada.pk}": 1,
        f"testapp.person:{bob.pk}": 1,
        f"testapp.person:{carol.pk}": 1,
    }
    # Each replaced as committed: its own name, then its mentees', if any.
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
        f"testapp.person:{bob.pk}": [
            NormalizedDocument(
                text="Bob\n\nAda",
                source_app_label="testapp",
                source_model="person",
                source_pk=bob.pk,
                title="Bob",
            ),
        ],
        f"testapp.person:{carol.pk}": [
            NormalizedDocument(
                text="Carol",
                source_app_label="testapp",
                source_model="person",
                source_pk=carol.pk,
                title="Carol",
            ),
        ],
    }


@pytest.mark.django_db
def test_adding_a_friend_to_a_friend_following_its_friends_resyncs_every_friend(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    _register_friends_following_their_friends()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the add below is observed. Carol is a friend
    # of no one, untouched by the add.
    ada = Friend.objects.create(name="Ada")
    bob = Friend.objects.create(name="Bob")
    carol = Friend.objects.create(name="Carol")

    # Added from Ada's side: Django sends m2m_changed with Ada as its instance
    # and Bob's primary key in pk_set, and writes the link back too, since the
    # many-to-many is symmetrical. No row is saved again: the add writes only
    # the links between them.
    with django_capture_on_commit_callbacks(execute=True):
        ada.friends.add(bob)
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The instance's group is replaced, and the other side is a Friend too,
    # read through the relation, so the commit resyncs every friend, the
    # untouched one included: each friend's group is replaced in exactly one
    # replace call, however the groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.friend:{ada.pk}": 1,
        f"testapp.friend:{bob.pk}": 1,
        f"testapp.friend:{carol.pk}": 1,
    }
    # Each replaced as committed: its own name, then its friend's, if any. Bob's
    # group shows Ada's name, through the link back.
    assert _received_groups(built_outputs) == {
        f"testapp.friend:{ada.pk}": [
            NormalizedDocument(
                text="Ada\n\nBob",
                source_app_label="testapp",
                source_model="friend",
                source_pk=ada.pk,
                title="Ada",
            ),
        ],
        f"testapp.friend:{bob.pk}": [
            NormalizedDocument(
                text="Bob\n\nAda",
                source_app_label="testapp",
                source_model="friend",
                source_pk=bob.pk,
                title="Bob",
            ),
        ],
        f"testapp.friend:{carol.pk}": [
            NormalizedDocument(
                text="Carol",
                source_app_label="testapp",
                source_model="friend",
                source_pk=carol.pk,
                title="Carol",
            ),
        ],
    }


@pytest.mark.django_db
def test_an_m2m_changed_sent_by_a_model_through_no_many_to_many_replaces_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the sender can keep the signal from
    # replacing anything.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Friend follows its friends: an add on its own many-to-many would replace
    # the groups of both sides.
    _register_friends_following_their_friends()

    # Created outside the captured callbacks: the commit callbacks of these
    # saves never run, so only the signal below is observed.
    bob = Friend.objects.create(name="Bob")
    ada = Friend.objects.create(name="Ada")

    # Sent by hand, as a project may, by Category: a plain model, the through
    # model of no many-to-many. Friend's many-to-many is not the one it is
    # about, nor any other: the signal changes no link the package can read.
    with django_capture_on_commit_callbacks(execute=True):
        m2m_changed.send(
            sender=Category,
            instance=ada,
            action="post_add",
            reverse=False,
            model=Friend,
            pk_set={bob.pk},
            using="default",
        )

    # Not even at the commit: no group of either Friend.
    assert _received_groups(built_outputs) == {}


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


@pytest.mark.django_db(transaction=True)
def test_a_category_an_offer_reads_saved_in_a_rolled_back_transaction_resyncs_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Created before Offer is registered: in autocommit (transaction=True),
    # their own commit callbacks would otherwise run, and send the offers'
    # groups.
    lighting = _create_an_offer_in_tools_and_in_lighting().lighting

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: a save of
    # the category schedules a resync of every offer for the commit.
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # A real transaction (transaction=True), so that leaving the atomic block
    # on an exception rolls it back rather than a savepoint of the test's own.
    with pytest.raises(_RolledBackError), transaction.atomic():
        lighting.name = "Lamps"
        lighting.save()
        raise _RolledBackError

    # No offer's group, nor any other: no replace call at all.
    assert _replaced(built_outputs) == []


@pytest.mark.django_db
def test_a_write_after_a_rolled_back_savepoint_that_scheduled_the_resync_resyncs_it(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: a save of a
    # category schedules a resync of every offer for the commit.
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the writes below are observed. One offer is in each
    # category.
    offers = _create_an_offer_in_tools_and_in_lighting()
    hammer_deal, lamp_sale = offers.hammer_deal, offers.lamp_sale
    tools, lighting = offers.tools, offers.lighting

    with django_capture_on_commit_callbacks(execute=True):
        # The first write the Offer reads happens in a savepoint (a nested
        # atomic block) that rolls back: the resync it scheduled is discarded
        # with it.
        with pytest.raises(_RolledBackError), transaction.atomic():
            lighting.name = "Lamps"
            lighting.save()
            raise _RolledBackError
        # A later write in the same transaction, outside the savepoint, to a
        # model the Offer reads.
        tools.name = "Hand tools"
        tools.save()
        # Nothing may reach the output before the commit.
        assert _replaced(built_outputs) == []

    # The later write still resyncs every offer at the commit, each once:
    # each offer's group is replaced in exactly one replace call, however the
    # groups are batched.
    replaced_source_keys = _replaced_source_keys(built_outputs)
    assert Counter(replaced_source_keys) == {
        f"testapp.offer:{hammer_deal.pk}": 1,
        f"testapp.offer:{lamp_sale.pk}": 1,
    }
    # Each replaced as committed: the rolled-back rename is not in it.
    assert _received_groups(built_outputs) == {
        f"testapp.offer:{hammer_deal.pk}": [
            NormalizedDocument(
                text="Hammer deal\n\nHand tools",
                source_app_label="testapp",
                source_model="offer",
                source_pk=hammer_deal.pk,
                title="Hammer deal",
            ),
        ],
        f"testapp.offer:{lamp_sale.pk}": [
            NormalizedDocument(
                text="Lamp sale\n\nLighting",
                source_app_label="testapp",
                source_model="offer",
                source_pk=lamp_sale.pk,
                title="Lamp sale",
            ),
        ],
    }


@pytest.mark.django_db
def test_a_category_an_offer_reads_saved_with_signals_off_resyncs_nothing(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    # A working output, so that only the setting can keep the save from
    # resyncing the offers.
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Only the Offer is registered, reading its product's category's name
    # through a lookup path two foreign keys deep, with no follow: with signals
    # on, a save of a category schedules a resync of every offer for the
    # commit. Category itself is not registered.
    _register_offers_reading_their_category_two_foreign_keys_deep()

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the category's save below is observed. One offer is in each
    # category.
    lighting = _create_an_offer_in_tools_and_in_lighting().lighting

    settings.MODEL_RAG_SIGNALS = False

    # The commit callbacks run: one resyncing the offers would build an output.
    with django_capture_on_commit_callbacks(execute=True):
        lighting.name = "Lamps"
        lighting.save()

    # The row is written, yet no offer's group, nor any other, is sent, even
    # after the commit: no output is built.
    assert Category.objects.get(pk=lighting.pk).name == "Lamps"
    assert built_outputs == []


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


@pytest.mark.django_db
def test_an_output_failing_on_one_resynced_model_still_receives_the_other_ones_groups(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    # Two registered models read the Category, neither following the other:
    # the Offer through a lookup path two foreign keys deep, registered first,
    # the Listing through a lookup path over its own foreign key, both with no
    # follow. Neither Category nor Product is registered.
    _register_offers_reading_their_category_two_foreign_keys_deep()
    rag.register(Listing, fields=["title", "category__name"])

    # Created outside the captured callbacks: their commit callbacks never
    # run, so only the category's save below is observed. Their source keys
    # are known before the output is configured to fail on one of them.
    lighting = _create_the_lighting_category()
    lamp = _create_a_plain_desk_lamp(lighting)
    lamp_sale = Offer.objects.create(title="Lamp sale", product=lamp)
    lamp_stand = Listing.objects.create(
        title="Lamp stand", link="/lamp-stand/", category=lighting
    )
    built_outputs.clear()

    # The output fails on the offer's group, so on the Offer's resync alone.
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": FAILING_ON_KEY_BACKEND,
        "OPTIONS": {"failing_source_key": f"testapp.offer:{lamp_sale.pk}"},
    }

    # An error escaping the commit callbacks would fail the test: the commit
    # itself must not raise.
    with (
        caplog.at_level(logging.ERROR, logger=PACKAGE_LOGGER),
        django_capture_on_commit_callbacks(execute=True),
    ):
        lighting.name = "Lamps"
        lighting.save()

    [record] = _package_log_records(caplog)
    assert record.levelno == logging.ERROR
    # The record names the model whose resync failed, and carries the error.
    assert "testapp.offer" in record.getMessage()
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], FailingReplaceError)
    # The Listing's resync, scheduled by the same commit, still reaches an
    # output, as committed.
    assert _received_groups(built_outputs) == {
        f"testapp.listing:{lamp_stand.pk}": [
            NormalizedDocument(
                text="Lamp stand\n\nLamps",
                source_app_label="testapp",
                source_model="listing",
                source_pk=lamp_stand.pk,
                title="Lamp stand",
            ),
        ],
    }


# How a query on Category compares a row's primary key with a parameter.
CATEGORY_PK_LOOKUP = '"testapp_category"."id" = %s'


def _reads_category_row(sql: str, params: Any, pk: Any) -> bool:
    """Whether ``sql`` is a SELECT looking up the Category row of primary key ``pk``.

    The parameter compared is the lookup's own, found by counting the
    placeholders before it: a query's other parameters (such as the constant
    exists() selects) may hold the same value without being the lookup's.
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
