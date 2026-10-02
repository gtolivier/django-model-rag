import pytest

from django_model_rag import BaseExtractor, NormalizedDocument, SyncPipeline, rag
from tests.testapp.models import Category


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
