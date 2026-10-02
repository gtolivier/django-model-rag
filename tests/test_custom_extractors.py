from collections.abc import Iterator

import pytest
from django.core.exceptions import ImproperlyConfigured

from django_model_rag import BaseExtractor, NormalizedDocument, SyncPipeline, rag
from tests.testapp.models import AccordionItem, Category, Page


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

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=f"Everything filed under {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    documents = SyncPipeline().run()

    assert [(document.source_pk, document.text) for document in documents] == [
        (tools.pk, "Everything filed under Tools."),
        (garden.pk, "Everything filed under Garden."),
        (kitchen.pk, "Everything filed under Kitchen."),
    ]


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


def test_register_extractor_gives_back_the_decorated_class_itself() -> None:
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=f"Everything filed under {instance.name}.",
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    # Applying the decorator by hand is what ``@rag.register_extractor(...)``
    # does to the class, and keeps a handle on the undecorated class.
    decorated = rag.register_extractor(Category)(CategoryExtractor)

    assert decorated is CategoryExtractor


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
        rag.register_extractor(Category)(StandaloneCategoryExtractor)  # type: ignore[type-var]


def test_registering_an_extractor_without_extract_fails() -> None:
    class UnfinishedCategoryExtractor(BaseExtractor[Category]):
        pass

    with pytest.raises(ImproperlyConfigured, match="UnfinishedCategoryExtractor"):
        # An extractor left abstract is the slip under test: the type checker
        # rightly rejects it.
        rag.register_extractor(Category)(UnfinishedCategoryExtractor)  # type: ignore[type-abstract]
