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
def test_document_text_leaves_out_a_declared_field_with_an_empty_value(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="", price="9.90", category=category
    )
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer"


@pytest.mark.django_db
def test_document_text_leaves_out_a_declared_field_with_a_blank_value(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="  \n ", price="9.90", category=category
    )
    register(Product, fields=["name", "description"])

    [document] = SyncPipeline().run()

    assert document.text == "Hammer"


@pytest.mark.django_db
def test_instance_without_text_in_its_declared_fields_produces_no_document(
    register: Callable[..., None],
) -> None:
    named = Category.objects.create(name="Tools")
    Category.objects.create(name="")
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.source_pk for document in documents] == [named.pk]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_documents_come_in_ascending_primary_key_order(
    register: Callable[..., None],
) -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    kitchen = Category.objects.create(name="Kitchen")
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.source_pk for document in documents] == [
        tools.pk,
        garden.pk,
        kitchen.pk,
    ]


@pytest.mark.django_db
def test_documents_of_several_models_come_grouped_in_registration_order(
    register: Callable[..., None],
) -> None:
    tools = Category.objects.create(name="Tools")
    garden = Category.objects.create(name="Garden")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=tools
    )
    rake = Product.objects.create(
        name="Rake", description="Gathers leaves.", price="14.50", category=garden
    )
    register(Product, fields=["name"])
    register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [(document.source_model, document.source_pk) for document in documents] == [
        ("product", hammer.pk),
        ("product", rake.pk),
        ("category", tools.pk),
        ("category", garden.pk),
    ]


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
def test_document_title_is_empty_when_the_first_declared_field_is_blank(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer", description="   ", price="9.90", category=category
    )
    register(Product, fields=["description", "name"])

    [document] = SyncPipeline().run()

    assert document.title == ""


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


@pytest.mark.django_db
def test_document_text_holds_the_string_form_of_a_non_text_field(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Drill", description="Bores holes.", price="149.00", category=category
    )
    register(Product, fields=["name", "price"])

    [document] = SyncPipeline().run()

    assert document.text == "Drill\n\n149.00"


@pytest.mark.django_db
def test_document_text_keeps_a_declared_field_whose_value_is_zero(
    register: Callable[..., None],
) -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Sticker", description="A free gift.", price="0", category=category
    )
    register(Product, fields=["name", "price"])

    [document] = SyncPipeline().run()

    assert document.text == "Sticker\n\n0.00"
