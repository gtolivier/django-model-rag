from collections.abc import Iterator

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_model_rag import (
    AlreadyRegistered,
    BaseExtractor,
    NormalizedDocument,
    SyncPipeline,
    rag,
)
from tests.testapp.models import AccordionItem, Category, Page, Product, TextPlugin


class CategoryExtractor(BaseExtractor[Category]):
    """Describe each category as "Everything filed under <its name>."."""

    def extract(self, instance: Category) -> NormalizedDocument:
        return NormalizedDocument(
            text=f"Everything filed under {instance.name}.",
            source_app_label="testapp",
            source_model="category",
            source_pk=instance.pk,
        )


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

    assert SyncPipeline().run() == [
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

    documents = SyncPipeline().run()

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

    first_run = [document.text for document in SyncPipeline().run()]
    second_run = [document.text for document in SyncPipeline().run()]

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

    assert SyncPipeline().run() == [
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

    assert SyncPipeline().run() == [
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

    assert SyncPipeline().run() == [
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

    assert SyncPipeline().run_instance(faq) == [
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
def test_extractor_returning_a_string_fails_naming_the_extractor() -> None:
    Category.objects.create(name="Tools")

    @rag.register_extractor(Category)
    class SloppyCategoryExtractor(BaseExtractor[Category]):
        # A bare string instead of a document is the slip under test: the
        # type checker rightly rejects it.
        def extract(self, instance: Category) -> str:  # type: ignore[override]
            return f"Everything filed under {instance.name}."

    with pytest.raises(TypeError, match="SloppyCategoryExtractor"):
        SyncPipeline().run()


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
        SyncPipeline().run()


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
        SyncPipeline().run()


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


@pytest.mark.django_db
def test_extractor_for_a_model_registered_with_fields_fails_and_keeps_them() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    with pytest.raises(AlreadyRegistered):
        rag.register_extractor(Category)(CategoryExtractor)

    [document] = SyncPipeline().run()
    assert document.text == "Tools"


@pytest.mark.django_db
def test_fields_for_a_model_with_an_extractor_fail_and_keep_the_extractor() -> None:
    Category.objects.create(name="Tools")
    rag.register_extractor(Category)(CategoryExtractor)

    with pytest.raises(AlreadyRegistered):
        rag.register(Category, fields=["name"])

    [document] = SyncPipeline().run()
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

    [document] = SyncPipeline().run()
    assert document.text == "Everything filed under Tools."


@pytest.mark.django_db
def test_unregistered_model_with_an_extractor_produces_no_document() -> None:
    Category.objects.create(name="Tools")
    rag.register_extractor(Category)(CategoryExtractor)

    rag.unregister(Category)

    assert SyncPipeline().run() == []


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

    documents = SyncPipeline().run()

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

    assert SyncPipeline().run() == [
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

    assert SyncPipeline().run() == [
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

    assert SyncPipeline().run() == [
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
