import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    Category,
    Page,
    Product,
    Review,
    Supplier,
    SupplierProfile,
    Workshop,
)


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


@pytest.mark.django_db
def test_lookup_path_keeps_its_place_in_the_declared_field_order() -> None:
    _create_product(name="Chair", category_name="Furniture")
    rag.register(Product, fields=["category__name", "name"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Furniture\n\nChair"]


@pytest.mark.django_db
def test_lookup_path_declared_first_gives_the_title() -> None:
    _create_product(name="Chair", category_name="Furniture")
    rag.register(Product, fields=["category__name", "name"])

    documents = SyncPipeline().run()

    assert [document.title for document in documents] == ["Furniture"]


@pytest.mark.django_db
def test_lookup_path_crossing_several_relations_adds_the_last_field_text() -> None:
    product = _create_product(name="Chair", category_name="Furniture")
    Review.objects.create(title="Sturdy", product=product)
    rag.register(Review, fields=["title", "product__category__name"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Sturdy\n\nFurniture"]


@pytest.mark.django_db
def test_lookup_path_ending_on_a_field_with_choices_gives_its_display_label() -> None:
    product = _create_product(name="Chair", category_name="Furniture")
    product.condition = "used"
    product.save()
    Review.objects.create(title="Sturdy", product=product)
    rag.register(Review, fields=["title", "product__condition"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Sturdy\n\nSecond-hand"]


@pytest.mark.django_db
def test_lookup_path_through_a_null_foreign_key_adds_nothing() -> None:
    Workshop.objects.create(title="Pottery", topic=None)
    rag.register(Workshop, fields=["title", "topic__title"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Pottery"]


@pytest.mark.django_db
def test_lookup_path_crossing_a_reverse_one_to_one_by_its_query_name() -> None:
    supplier = Supplier.objects.create(name="Acme")
    SupplierProfile.objects.create(supplier=supplier, body="Fine tools since 1920")
    rag.register(Supplier, fields=["name", "supplier_profile__body"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Acme\n\nFine tools since 1920"
    ]


@pytest.mark.django_db
def test_lookup_path_through_a_missing_reverse_one_to_one_adds_nothing() -> None:
    Page.objects.create(title="Home", slug="home")
    rag.register(Page, fields=["title", "intro__body"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Home"]
