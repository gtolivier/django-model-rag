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
