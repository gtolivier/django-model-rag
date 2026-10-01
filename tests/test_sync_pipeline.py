from collections.abc import Callable

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


@pytest.mark.django_db
def test_registered_model_produces_a_document_from_its_declared_field(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Hammer"]


@pytest.mark.django_db
def test_document_text_joins_declared_fields_with_a_blank_line(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer\n\nDrives nails."


@pytest.mark.django_db
def test_document_text_follows_the_declared_field_order(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.text == "Drives nails.\n\nHammer"


@pytest.mark.django_db
def test_document_title_is_the_value_of_the_first_declared_field(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.title == "Hammer"


@pytest.mark.django_db
def test_document_title_follows_the_declared_field_order(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.title == "Drives nails."


@pytest.mark.django_db
def test_document_carries_the_source_of_its_product(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    product = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    register(Product, fields=["name"])

    [document] = SyncPipeline().run()

    assert (
        document.source_app_label,
        document.source_model,
        document.source_pk,
    ) == ("testapp", "product", product.pk)


@pytest.mark.django_db
def test_document_carries_the_source_of_its_own_model(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(pk=42, name="Tools")
    register(Category, fields=["name"])

    [document] = SyncPipeline().run()

    assert (
        document.source_app_label,
        document.source_model,
        document.source_pk,
    ) == ("testapp", "category", category.pk)
