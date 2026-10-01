import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Category, Product


def test_pipeline_without_registered_model_produces_no_document() -> None:
    assert SyncPipeline().run() == []


@pytest.mark.django_db
def test_unregistered_model_produces_no_document() -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    rag.register(Product, fields=["name"])
    rag.unregister(Product)

    assert SyncPipeline().run() == []
