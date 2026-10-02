from django_model_rag import BaseExtractor, NormalizedDocument, rag
from tests.testapp.models import Category, Page, Product


def test_registered_models_is_empty_when_no_model_is_registered() -> None:
    assert list(rag.registered_models()) == []


def test_registered_models_lists_models_in_registration_order_across_kinds() -> None:
    rag.register(Product, fields=["name"])

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=instance.name,
                source_app_label="testapp",
                source_model="category",
                source_pk=instance.pk,
            )

    rag.register(Page, fields=["title"])

    assert rag.registered_models() == [Product, Category, Page]
