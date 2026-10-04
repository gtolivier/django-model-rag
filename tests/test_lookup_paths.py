import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Category, Product


def _create_product(*, name: str, category_name: str) -> Product:
    """Create a product called ``name`` in a new category called ``category_name``."""
    category = Category.objects.create(name=category_name)
    return Product.objects.create(
        name=name, description="", price="9.90", category=category
    )


@pytest.mark.django_db
def test_foreign_key_lookup_path_adds_each_instance_its_related_field_text() -> None:
    _create_product(name="Chair", category_name="Furniture")
    _create_product(name="Hammer", category_name="Tools")
    rag.register(Product, fields=["name", "category__name"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nFurniture",
        "Hammer\n\nTools",
    ]


@pytest.mark.django_db
def test_run_instance_adds_the_related_field_text_of_a_lookup_path() -> None:
    product = _create_product(name="Chair", category_name="Furniture")
    rag.register(Product, fields=["name", "category__name"])

    documents = SyncPipeline().run_instance(product)

    assert [document.text for document in documents] == ["Chair\n\nFurniture"]
