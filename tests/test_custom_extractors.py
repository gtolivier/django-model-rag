import subprocess
import sys
import textwrap
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Model, QuerySet
from pytest_django import DjangoAssertNumQueries

from django_model_rag import (
    AlreadyRegistered,
    BaseExtractor,
    NormalizedDocument,
    rag,
)
from tests.recording import PRUNE_KEYS_QUERY, run_documents, run_instance_documents
from tests.testapp.models import (
    AccordionItem,
    Category,
    Course,
    Membership,
    Page,
    Photo,
    Pin,
    Product,
    Tag,
    TextPlugin,
    Topic,
)


class CategoryExtractor(BaseExtractor[Category]):
    """Describe each category as "Everything filed under <its name>."."""

    def extract(self, instance: Category) -> NormalizedDocument:
        return NormalizedDocument(
            text=f"Everything filed under {instance.name}.",
            source_app_label="testapp",
            source_model="category",
            source_pk=instance.pk,
        )


class TextPluginBodyExtractor(BaseExtractor[TextPlugin]):
    """Describe each text plugin by its body."""

    def extract(self, instance: TextPlugin) -> NormalizedDocument:
        return self.build_document(instance, text=instance.body)


class AnyModelExtractor(BaseExtractor[Model]):
    """Describe an instance of any model by its str()."""

    def extract(self, instance: Model) -> NormalizedDocument:
        return self.build_document(instance, text=str(instance))


@pytest.mark.django_db
def test_registered_extractor_builds_the_document_of_its_model() -> None:
    category = Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=f"Everything filed under {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
                title=f"The {instance.name} category",
            )

    assert run_documents() == [
        NormalizedDocument(
            text="Everything filed under Tools.",
            source_app_label="testapp",
            source_model="category",
            source_pk=category.pk,
            title="The Tools category",
        )
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_registered_extractor_builds_one_document_per_instance_in_pk_order() -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    kitchen = Category.objects.create(name="Kitchen")
    rag.register_extractor(Category)(CategoryExtractor)

    documents = run_documents()

    assert [(document.source_pk, document.text) for document in documents] == [
        (tools.pk, "Everything filed under Tools."),
        (garden.pk, "Everything filed under Garden."),
        (kitchen.pk, "Everything filed under Kitchen."),
    ]


@pytest.mark.django_db
def test_each_run_starts_with_a_fresh_extractor() -> None:
    Category.objects.create(name="Tools")
    Category.objects.create(name="Garden")

    @rag.register_extractor(Category)
    class NumberingCategoryExtractor(BaseExtractor[Category]):
        # State kept on self during a run is what is under test: an extractor
        # shared across runs would keep counting from the first run's total.
        def __init__(self) -> None:
            self.built = 0

        def extract(self, instance: Category) -> NormalizedDocument:
            self.built += 1
            return self.build_document(
                instance, text=f"Category {self.built}: {instance.name}."
            )

    first_run = [document.text for document in run_documents()]
    second_run = [document.text for document in run_documents()]

    assert first_run == ["Category 1: Tools.", "Category 2: Garden."]
    assert second_run == first_run


@pytest.mark.django_db
def test_instance_the_extractor_returns_none_for_produces_no_document() -> None:
    tools = Category.objects.create(name="Tools")
    Category.objects.create(name="   ")
    kitchen = Category.objects.create(name="Kitchen")

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument | None:
            if not instance.name.strip():
                return None
            return NormalizedDocument(
                text=f"Everything filed under {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    assert run_documents() == [
        NormalizedDocument(
            text="Everything filed under Tools.",
            source_app_label="testapp",
            source_model="category",
            source_pk=tools.pk,
        ),
        NormalizedDocument(
            text="Everything filed under Kitchen.",
            source_app_label="testapp",
            source_model="category",
            source_pk=kitchen.pk,
        ),
    ]


@pytest.mark.django_db
def test_extractor_may_return_several_documents_for_one_instance() -> None:
    page = Page.objects.create(title="FAQ", slug="faq")
    AccordionItem.objects.create(
        page=page, title="Shipping", body="We ship within two days."
    )
    AccordionItem.objects.create(
        page=page, title="Returns", body="Returns are free for thirty days."
    )

    @rag.register_extractor(Page)
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> list[NormalizedDocument]:
            return [
                NormalizedDocument(
                    text=item.body,
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=instance.pk,
                    title=item.title,
                )
                for item in instance.accordion_items.order_by("pk")
            ]

    assert run_documents() == [
        NormalizedDocument(
            text="We ship within two days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=page.pk,
            title="Shipping",
        ),
        NormalizedDocument(
            text="Returns are free for thirty days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=page.pk,
            title="Returns",
        ),
    ]


@pytest.mark.django_db
def test_extractor_may_yield_its_documents_for_one_instance() -> None:
    page = Page.objects.create(title="FAQ", slug="faq")
    AccordionItem.objects.create(
        page=page, title="Shipping", body="We ship within two days."
    )
    AccordionItem.objects.create(
        page=page, title="Returns", body="Returns are free for thirty days."
    )

    @rag.register_extractor(Page)
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> Iterator[NormalizedDocument]:
            for item in instance.accordion_items.order_by("pk"):
                yield NormalizedDocument(
                    text=item.body,
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=instance.pk,
                    title=item.title,
                )

    assert run_documents() == [
        NormalizedDocument(
            text="We ship within two days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=page.pk,
            title="Shipping",
        ),
        NormalizedDocument(
            text="Returns are free for thirty days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=page.pk,
            title="Returns",
        ),
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_extractor_get_queryset_shapes_the_queryset_its_instances_are_read_from(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three pages with two text plugins each: without the prefetch, one query
    # per page would show as more than two queries, and the reversed selects
    # check that the pages still come in primary key order.
    about = Page.objects.create(title="About us", slug="about-us")
    contact = Page.objects.create(title="Contact", slug="contact")
    visits = Page.objects.create(title="Visits", slug="visits")
    TextPlugin.objects.create(page=about, body="We build chairs by hand.")
    TextPlugin.objects.create(page=contact, body="Write to us.")
    TextPlugin.objects.create(page=visits, body="Visits on Saturdays.")
    TextPlugin.objects.create(page=about, body="Our workshop is in Lyon.")
    TextPlugin.objects.create(page=contact, body="Or call us.")
    TextPlugin.objects.create(page=visits, body="Book a week ahead.")

    @rag.register_extractor(Page)
    class PageExtractor(BaseExtractor[Page]):
        def get_queryset(self, queryset: QuerySet[Page]) -> QuerySet[Page]:
            return queryset.prefetch_related("text_plugins")

        def extract(self, instance: Page) -> NormalizedDocument:
            # sorted in Python, not order_by(): the plugins stay prefetched
            plugins = sorted(instance.text_plugins.all(), key=lambda plugin: plugin.pk)
            return self.build_document(
                instance, text=" ".join(plugin.body for plugin in plugins)
            )

    with django_assert_num_queries(2 + PRUNE_KEYS_QUERY):
        documents = run_documents()

    assert [(document.source_pk, document.text) for document in documents] == [
        (about.pk, "We build chairs by hand. Our workshop is in Lyon."),
        (contact.pk, "Write to us. Or call us."),
        (visits.pk, "Visits on Saturdays. Book a week ahead."),
    ]


@pytest.mark.django_db
def test_extractor_get_queryset_may_select_the_related_objects_extract_reads(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three products in two categories: without the join, reading each
    # product's category would show as more than one query.
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )
    rake = Product.objects.create(
        name="Rake", description="Gathers leaves.", price="14.50", category=garden
    )
    saw = Product.objects.create(
        name="Saw", description="Cuts planks.", price="19.90", category=tools
    )

    @rag.register_extractor(Product)
    class ProductExtractor(BaseExtractor[Product]):
        def get_queryset(self, queryset: QuerySet[Product]) -> QuerySet[Product]:
            return queryset.select_related("category")

        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"{instance.name}, filed under {instance.category.name}."
            )

    with django_assert_num_queries(1 + PRUNE_KEYS_QUERY):
        documents = run_documents()

    assert [(document.source_pk, document.text) for document in documents] == [
        (hammer.pk, "Hammer, filed under Tools."),
        (rake.pk, "Rake, filed under Garden."),
        (saw.pk, "Saw, filed under Tools."),
    ]


@pytest.mark.django_db
def test_extractor_get_queryset_ordering_otherwise_keeps_documents_in_pk_order() -> (
    None
):
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    kitchen = Category.objects.create(name="Kitchen")

    @rag.register_extractor(Category)
    class NewestFirstCategoryExtractor(BaseExtractor[Category]):
        # An ordering of the extractor's own is the slip under test:
        # get_queryset() shapes how the instances load, not their order.
        def get_queryset(self, queryset: QuerySet[Category]) -> QuerySet[Category]:
            return queryset.order_by("-pk")

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    documents = run_documents()

    assert [(document.source_pk, document.text) for document in documents] == [
        (tools.pk, "Tools"),
        (garden.pk, "Garden"),
        (kitchen.pk, "Kitchen"),
    ]


@pytest.mark.django_db
def test_extractor_attribute_named_like_a_registration_option_shapes_nothing(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    tools = Category.objects.create(name="Tools")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )

    @rag.register_extractor(Product)
    class ProductExtractor(BaseExtractor[Product]):
        # An attribute of the extractor's own that happens to be called
        # ``follow``: only get_queryset() may shape the queryset, and extract()
        # reads nothing of the category it names.
        follow = ("category",)

        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    with django_assert_num_queries(1 + PRUNE_KEYS_QUERY) as queries:
        documents = run_documents()

    assert [(document.source_pk, document.text) for document in documents] == [
        (hammer.pk, "Hammer")
    ]
    assert Category._meta.db_table not in queries.captured_queries[0]["sql"]


@pytest.mark.django_db
def test_run_instance_produces_the_extracted_documents_of_that_instance_only() -> None:
    faq = Page.objects.create(title="FAQ", slug="faq")
    AccordionItem.objects.create(
        page=faq, title="Shipping", body="We ship within two days."
    )
    AccordionItem.objects.create(
        page=faq, title="Returns", body="Returns are free for thirty days."
    )
    about = Page.objects.create(title="About", slug="about")
    AccordionItem.objects.create(
        page=about, title="History", body="Founded in a garage."
    )

    @rag.register_extractor(Page)
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> list[NormalizedDocument]:
            return [
                self.build_document(instance, text=item.body, title=item.title)
                for item in instance.accordion_items.order_by("pk")
            ]

    assert run_instance_documents(faq) == [
        NormalizedDocument(
            text="We ship within two days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=faq.pk,
            title="Shipping",
        ),
        NormalizedDocument(
            text="Returns are free for thirty days.",
            source_app_label="testapp",
            source_model="page",
            source_pk=faq.pk,
            title="Returns",
        ),
    ]


@pytest.mark.django_db
def test_each_run_instance_starts_with_a_fresh_extractor() -> None:
    faq = Page.objects.create(title="FAQ", slug="faq")
    AccordionItem.objects.create(
        page=faq, title="Shipping", body="We ship within two days."
    )
    AccordionItem.objects.create(
        page=faq, title="Returns", body="Returns are free for thirty days."
    )

    @rag.register_extractor(Page)
    class NumberingPageExtractor(BaseExtractor[Page]):
        # State kept on self during a call is what is under test: an extractor
        # shared across calls would keep counting from the first call's total.
        def __init__(self) -> None:
            self.built = 0

        def extract(self, instance: Page) -> list[NormalizedDocument]:
            documents = []
            for item in instance.accordion_items.order_by("pk"):
                self.built += 1
                documents.append(
                    self.build_document(instance, text=f"{self.built}. {item.title}")
                )
            return documents

    first_call = [document.text for document in run_instance_documents(faq)]
    second_call = [document.text for document in run_instance_documents(faq)]

    assert first_call == ["1. Shipping", "2. Returns"]
    assert second_call == first_call


@pytest.mark.django_db
def test_extractor_returning_a_string_fails_naming_the_extractor() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class SloppyCategoryExtractor(BaseExtractor[Category]):
        # A bare string instead of a document is the slip under test: the
        # type checker rightly rejects it.
        def extract(self, instance: Category) -> str:  # type: ignore[override]
            return f"Everything filed under {instance.name}."

    with pytest.raises(TypeError, match="SloppyCategoryExtractor"):
        run_documents()


@pytest.mark.django_db
def test_extractor_returning_a_non_iterable_fails_naming_the_extractor() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class CountingCategoryExtractor(BaseExtractor[Category]):
        # A number instead of a document is the slip under test: the type
        # checker rightly rejects it.
        def extract(self, instance: Category) -> int:  # type: ignore[override]
            return instance.pk

    with pytest.raises(TypeError, match="CountingCategoryExtractor"):
        run_documents()


@pytest.mark.django_db
def test_extractor_yielding_something_else_than_a_document_fails_naming_it() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class StringListCategoryExtractor(BaseExtractor[Category]):
        # A list of strings instead of documents is the slip under test: the
        # type checker rightly rejects it.
        def extract(self, instance: Category) -> list[str]:  # type: ignore[override]
            return [f"Everything filed under {instance.name}."]

    with pytest.raises(TypeError, match="StringListCategoryExtractor"):
        run_documents()


@pytest.mark.django_db
def test_extractor_get_queryset_returning_a_list_fails_naming_the_extractor() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class ListingCategoryExtractor(BaseExtractor[Category]):
        # A list of the instances instead of a queryset is the slip under
        # test: the type checker rightly rejects it.
        def get_queryset(  # type: ignore[override]
            self, queryset: QuerySet[Category]
        ) -> list[Category]:
            return list(queryset)

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    with pytest.raises(
        TypeError, match=r"ListingCategoryExtractor\.get_queryset\(\).*QuerySet"
    ):
        run_documents()


@pytest.mark.django_db
def test_extractor_get_queryset_returning_values_fails_naming_the_extractor() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class ValuesCategoryExtractor(BaseExtractor[Category]):
        # A values() queryset, which loads dictionaries instead of instances,
        # is the slip under test: the type checker rightly rejects it.
        def get_queryset(  # type: ignore[override]
            self, queryset: QuerySet[Category]
        ) -> QuerySet[Category, Mapping[str, object]]:
            return queryset.values("pk", "name")

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text="A category.")

    with pytest.raises(TypeError, match=r"ValuesCategoryExtractor\.get_queryset\(\)"):
        run_documents()


@pytest.mark.django_db
def test_extractor_get_queryset_returning_another_model_fails_naming_both() -> None:
    tools = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )

    @rag.register_extractor(Category)
    class StrayExtractor(BaseExtractor[Category]):
        # A queryset of products for a category extractor is the slip under
        # test: the type checker rightly rejects it.
        def get_queryset(  # type: ignore[override]
            self, queryset: QuerySet[Category]
        ) -> QuerySet[Product]:
            return Product.objects.all()

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    with pytest.raises(TypeError, match=r"StrayExtractor\.get_queryset\(\).*Category"):
        run_documents()


def test_register_extractor_gives_back_the_decorated_class_itself() -> None:
    # Applying the decorator by hand is what ``@rag.register_extractor(...)``
    # does to the class, and keeps a handle on the undecorated class.
    decorated = rag.register_extractor(Category)(CategoryExtractor)

    assert decorated is CategoryExtractor


def test_type_checker_rejects_an_extractor_registered_for_another_model() -> None:
    class ProductExtractor(BaseExtractor[Product]):
        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(instance, text=instance.description)

    # A product extractor registered for categories is the slip under test:
    # the type checker rightly rejects it, before anything runs.
    decorated = rag.register_extractor(Category)(ProductExtractor)  # type: ignore[arg-type]

    assert decorated is ProductExtractor


def test_registering_a_class_not_derived_from_base_extractor_fails() -> None:
    class StandaloneCategoryExtractor:
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=f"Everything filed under {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    with pytest.raises(ImproperlyConfigured, match="StandaloneCategoryExtractor"):
        # A class that only looks like an extractor is the slip under test:
        # the type checker rightly rejects it.
        rag.register_extractor(Category)(StandaloneCategoryExtractor)  # type: ignore[arg-type]


def test_registering_a_function_instead_of_a_class_fails_naming_it() -> None:
    def extract_category(instance: Category) -> NormalizedDocument:
        return NormalizedDocument(
            text=f"Everything filed under {instance.name}.",
            source_app_label="testapp",
            source_model="category",
            source_pk=instance.pk,
        )

    with pytest.raises(ImproperlyConfigured, match="extract_category"):
        # A plain function instead of an extractor class is the slip under
        # test: the type checker rightly rejects it.
        rag.register_extractor(Category)(extract_category)  # type: ignore[arg-type]


def test_registering_an_extractor_without_extract_fails() -> None:
    class UnfinishedCategoryExtractor(BaseExtractor[Category]):
        pass

    with pytest.raises(ImproperlyConfigured, match="UnfinishedCategoryExtractor"):
        # An extractor left abstract is the slip under test: the type checker
        # rightly rejects it.
        rag.register_extractor(Category)(UnfinishedCategoryExtractor)  # type: ignore[type-abstract]


def test_depending_on_a_single_relation_name_instead_of_a_list_fails() -> None:
    with pytest.raises(
        ImproperlyConfigured, match=r"\bdepends_on\b.*\blist or a tuple\b"
    ):
        # A bare string is the slip under test: the type checker rightly
        # rejects it.
        rag.register_extractor(TextPlugin, depends_on="page")(TextPluginBodyExtractor)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "item",
    [
        pytest.param(None, id="none"),
        # The field object of TextPlugin.page, where its name was meant.
        pytest.param(TextPlugin.page.field, id="field-object"),
    ],
)
def test_depending_on_an_item_that_is_not_a_name_fails_at_registration(
    item: object,
) -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bdepends_on\b"):
        # An item that is not a string is the slip under test: the type
        # checker rightly rejects it.
        rag.register_extractor(TextPlugin, depends_on=[item])(TextPluginBodyExtractor)  # type: ignore[list-item]


@pytest.mark.parametrize(
    "name",
    [
        # TextPlugin.body exists but holds content: its changes are the
        # model's own, not a relation's.
        pytest.param("body", id="content-field"),
        pytest.param("pgae", id="unknown-name"),
    ],
)
def test_depending_on_a_name_that_is_not_a_relation_fails_at_registration(
    name: str,
) -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(TextPlugin, depends_on=[name])(TextPluginBodyExtractor)


@pytest.mark.parametrize(
    ("model", "name"),
    [
        pytest.param(Course, "topics", id="forward"),
        pytest.param(Topic, "courses", id="reverse"),
    ],
)
def test_depending_on_a_many_to_many_fails_at_registration(
    model: type[Model], name: str
) -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(model, depends_on=[name])(AnyModelExtractor)


@pytest.mark.parametrize(
    ("model", "name"),
    [
        # Tag.content_object may point to an instance of any model: there is
        # no single model whose changes to listen to.
        pytest.param(Tag, "content_object", id="generic-foreign-key"),
        pytest.param(Photo, "tags", id="generic-relation"),
    ],
)
def test_depending_on_a_generic_relation_fails_at_registration(
    model: type[Model], name: str
) -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(model, depends_on=[name])(AnyModelExtractor)


def test_depending_on_a_path_through_a_later_reverse_relation_fails() -> None:
    # TextPlugin.page is a foreign key, but Page.accordion_items is a reverse
    # one: past the first link, a path goes through foreign keys only.
    name = "page__accordion_items"
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(TextPlugin, depends_on=[name])(TextPluginBodyExtractor)


def test_depending_on_a_longer_path_starting_with_a_reverse_relation_fails() -> None:
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> NormalizedDocument:
            return self.build_document(instance, text=instance.title)

    # Page.text_plugins alone is allowed, but a reverse relation is a path's
    # last link only: a longer path goes through foreign keys only.
    name = "text_plugins__page"
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(Page, depends_on=[name])(PageExtractor)


def test_depending_on_a_path_through_a_later_non_relation_fails() -> None:
    # TextPlugin.page is a foreign key, but Page.title holds content: every
    # link of a path is a relation.
    name = "page__title"
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(TextPlugin, depends_on=[name])(TextPluginBodyExtractor)


def test_depending_on_a_path_through_a_later_unknown_name_fails_unregistered() -> None:
    # TextPlugin.page is a foreign key, but Page has no field named nope.
    name = "page__nope"
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(TextPlugin, depends_on=[name])(TextPluginBodyExtractor)

    assert TextPlugin not in rag.registered_models()


@pytest.mark.parametrize(
    ("model", "name"),
    [
        # Membership.guild is a foreign key, but Guild.members a many-to-many.
        pytest.param(Membership, "guild__members", id="many-to-many"),
        # Pin.tag is a foreign key, but Tag.content_object may point to an
        # instance of any model.
        pytest.param(Pin, "tag__content_object", id="generic-foreign-key"),
        # Pin.photo is a foreign key, but Photo.tags a generic relation.
        pytest.param(Pin, "photo__tags", id="generic-relation"),
    ],
)
def test_depending_on_a_path_through_a_later_many_to_many_or_generic_relation_fails(
    model: type[Model], name: str
) -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=rf"\b{name}\b.*\bdepends_on\b|\bdepends_on\b.*\b{name}\b",
    ):
        rag.register_extractor(model, depends_on=[name])(AnyModelExtractor)


def test_depending_on_a_relation_twice_names_it_in_the_error() -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=r"\bpage\b.*\btwice\b|\btwice\b.*\bpage\b",
    ):
        rag.register_extractor(TextPlugin, depends_on=["page", "page"])(
            TextPluginBodyExtractor
        )


THROWAWAY_APP_SETTINGS = """\
import django
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

settings.configure(
    INSTALLED_APPS=["noticeboard"],
    DEFAULT_AUTO_FIELD="django.db.models.AutoField",
)
"""
"""The start of a script that configures the throwaway ``noticeboard`` app."""


def run_with_throwaway_app(
    tmp_path: Path, models_source: str, script_source: str
) -> subprocess.CompletedProcess[str]:
    """Run ``script_source`` in a fresh interpreter, next to a throwaway app.

    The app, ``noticeboard``, has ``models_source`` as its models.py. This
    process's apps are long loaded: only a fresh interpreter runs a models.py
    while models are loading.
    """
    app = tmp_path / "noticeboard"
    app.mkdir()
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(textwrap.dedent(models_source))
    script = tmp_path / "load_apps.py"
    script.write_text(THROWAWAY_APP_SETTINGS + textwrap.dedent(script_source))
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_depending_on_a_relation_from_a_models_module_fails_and_points_to_ready(
    tmp_path: Path,
) -> None:
    # Resolving depends_on needs every model loaded, which is not the case
    # while a models.py runs.
    result = run_with_throwaway_app(
        tmp_path,
        """\
        from django.db import models

        from django_model_rag import BaseExtractor, rag


        class Board(models.Model):
            name = models.CharField(max_length=100)


        class Memo(models.Model):
            body = models.TextField()
            board = models.ForeignKey(Board, on_delete=models.CASCADE)


        @rag.register_extractor(Memo, depends_on=["board"])
        class MemoExtractor(BaseExtractor[Memo]):
            def extract(self, instance):
                return self.build_document(instance, text=instance.body)
        """,
        """\
        try:
            django.setup()
        except ImproperlyConfigured as error:
            print(error)
        """,
    )

    assert result.stdout, f"no ImproperlyConfigured raised; stderr:\n{result.stderr}"
    assert "rag.py" in result.stdout, result.stdout
    assert "AppConfig.ready()" in result.stdout, result.stdout


def test_registering_an_extractor_from_a_models_module_without_depends_on_works(
    tmp_path: Path,
) -> None:
    result = run_with_throwaway_app(
        tmp_path,
        """\
        from django.db import models

        from django_model_rag import BaseExtractor, rag


        class Memo(models.Model):
            body = models.TextField()


        @rag.register_extractor(Memo)
        class MemoExtractor(BaseExtractor[Memo]):
            def extract(self, instance):
                return self.build_document(instance, text=instance.body)
        """,
        """\
        django.setup()

        from django_model_rag import rag
        from noticeboard.models import Memo

        print(rag.is_registered(Memo))
        """,
    )

    assert result.returncode == 0, f"setup failed; stderr:\n{result.stderr}"
    assert result.stdout == "True\n", result.stdout


@pytest.mark.django_db
def test_extractor_for_a_model_registered_with_fields_fails_and_keeps_them() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    with pytest.raises(AlreadyRegistered):
        rag.register_extractor(Category)(CategoryExtractor)

    [document] = run_documents()
    assert document.text == "Tools"


@pytest.mark.django_db
def test_fields_for_a_model_with_an_extractor_fail_and_keep_the_extractor() -> None:
    Category.objects.create(name="Tools")
    rag.register_extractor(Category)(CategoryExtractor)

    with pytest.raises(AlreadyRegistered):
        rag.register(Category, fields=["name"])

    [document] = run_documents()
    assert document.text == "Everything filed under Tools."


@pytest.mark.django_db
def test_second_extractor_for_a_model_fails_and_keeps_the_first() -> None:
    Category.objects.create(name="Tools")
    rag.register_extractor(Category)(CategoryExtractor)

    class OtherCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=f"Another take on {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    with pytest.raises(AlreadyRegistered):
        rag.register_extractor(Category)(OtherCategoryExtractor)

    [document] = run_documents()
    assert document.text == "Everything filed under Tools."


def test_extractor_for_a_registered_model_fails_registered_before_depends_on() -> None:
    rag.register_extractor(Category)(CategoryExtractor)

    with pytest.raises(AlreadyRegistered):
        rag.register_extractor(Category, depends_on=["no_such_field"])(
            CategoryExtractor
        )


@pytest.mark.django_db
def test_unregistered_model_with_an_extractor_produces_no_document() -> None:
    Category.objects.create(name="Tools")
    rag.register_extractor(Category)(CategoryExtractor)

    rag.unregister(Category)

    assert run_documents() == []


@pytest.mark.django_db
def test_documents_come_grouped_in_registration_order_whatever_its_kind() -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )
    rake = Product.objects.create(
        name="Rake", description="Gathers leaves.", price="14.50", category=garden
    )
    faq = Page.objects.create(title="FAQ", slug="faq")
    about = Page.objects.create(title="About", slug="about")

    # Fields, then an extractor, then fields again: grouping the models by
    # kind of registration, in either order, would get it wrong.
    rag.register(Product, fields=["name"])
    rag.register_extractor(Category)(CategoryExtractor)
    rag.register(Page, fields=["title"])

    documents = run_documents()

    assert [(document.source_model, document.source_pk) for document in documents] == [
        ("product", hammer.pk),
        ("product", rake.pk),
        ("category", tools.pk),
        ("category", garden.pk),
        ("page", faq.pk),
        ("page", about.pk),
    ]


@pytest.mark.django_db
def test_build_document_takes_the_source_from_the_instance() -> None:
    category = Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(
                instance, text=f"Everything filed under {instance.name}."
            )

    assert run_documents() == [
        NormalizedDocument(
            text="Everything filed under Tools.",
            source_app_label="testapp",
            source_model="category",
            source_pk=category.pk,
        )
    ]


@pytest.mark.django_db
def test_build_document_takes_the_source_from_the_instance_of_any_model() -> None:
    page = Page.objects.create(title="FAQ", slug="faq")
    # Two plugins on one page: the second one's key cannot match the page's,
    # so a source taken from the page instead of the plugin would show.
    shipping = TextPlugin.objects.create(page=page, body="We ship within two days.")
    returns = TextPlugin.objects.create(
        page=page, body="Returns are free for thirty days."
    )

    @rag.register_extractor(TextPlugin)
    class TextPluginExtractor(BaseExtractor[TextPlugin]):
        def extract(self, instance: TextPlugin) -> NormalizedDocument:
            return self.build_document(instance, text=instance.body)

    assert run_documents() == [
        NormalizedDocument(
            text="We ship within two days.",
            source_app_label="testapp",
            source_model="textplugin",
            source_pk=shipping.pk,
        ),
        NormalizedDocument(
            text="Returns are free for thirty days.",
            source_app_label="testapp",
            source_model="textplugin",
            source_pk=returns.pk,
        ),
    ]


@pytest.mark.django_db
def test_build_document_passes_on_title_url_language_and_metadata() -> None:
    page = Page.objects.create(title="FAQ", slug="faq")
    plugin = TextPlugin.objects.create(page=page, body="We ship within two days.")

    @rag.register_extractor(TextPlugin)
    class TextPluginExtractor(BaseExtractor[TextPlugin]):
        def extract(self, instance: TextPlugin) -> NormalizedDocument:
            return self.build_document(
                instance,
                text=instance.body,
                title=instance.page.title,
                url=instance.page.get_absolute_url(),
                language="en",
                metadata={"plugin_type": "text", "page_id": instance.page.pk},
            )

    assert run_documents() == [
        NormalizedDocument(
            text="We ship within two days.",
            source_app_label="testapp",
            source_model="textplugin",
            source_pk=plugin.pk,
            title="FAQ",
            url="/pages/faq/",
            language="en",
            metadata={"plugin_type": "text", "page_id": page.pk},
        )
    ]


@pytest.mark.django_db
def test_build_document_gives_the_permissions_it_is_passed_or_none_as_frozenset() -> (
    None
):
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    public = Category.objects.create(name="Public")

    @rag.register_extractor(Category)
    class RestrictedCategoryExtractor(BaseExtractor[Category]):
        # Permissions derived from each instance, and left out for one: a
        # single value given to every document would not match.
        def extract(self, instance: Category) -> NormalizedDocument:
            if instance.name == "Public":
                return self.build_document(instance, text=instance.name)
            return self.build_document(
                instance,
                text=instance.name,
                permissions=[
                    f"testapp.view_{instance.name.lower()}",
                    "testapp.view_category",
                ],
            )

    documents = run_documents()

    assert [(document.source_pk, document.permissions) for document in documents] == [
        (tools.pk, frozenset({"testapp.view_tools", "testapp.view_category"})),
        (garden.pk, frozenset({"testapp.view_garden", "testapp.view_category"})),
        (public.pk, frozenset()),
    ]
    assert all(type(document.permissions) is frozenset for document in documents)


@pytest.mark.django_db
def test_build_document_refuses_a_bare_string_as_permissions() -> None:
    tools = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )

    @rag.register_extractor(Product)
    class SinglePermissionProductExtractor(BaseExtractor[Product]):
        # One permission given as a bare string instead of a collection is
        # the slip under test: read as an iterable, it would grant its
        # characters.
        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(
                instance, text=instance.name, permissions="testapp.view_product"
            )

    with pytest.raises(TypeError, match="permissions"):
        run_documents()


@pytest.mark.django_db
def test_extractor_keeping_fields_of_its_own_that_resolve_to_nothing_runs() -> None:
    tools = Category.objects.create(name="Tools")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )

    @rag.register_extractor(Product)
    class SummaryProductExtractor(BaseExtractor[Product]):
        # An attribute of the extractor's own that happens to be called
        # ``fields``: Product has no field ``summary`` for it to resolve to.
        fields = ("summary__short",)

        def extract(self, instance: Product) -> NormalizedDocument:
            return self.build_document(instance, text=f"In short: {instance.name}.")

    assert run_documents() == [
        NormalizedDocument(
            text="In short: Hammer.",
            source_app_label="testapp",
            source_model="product",
            source_pk=hammer.pk,
        )
    ]
