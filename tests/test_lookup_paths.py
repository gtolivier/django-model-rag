import re

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Model
from pytest_django import DjangoAssertNumQueries

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    Category,
    Course,
    Lesson,
    Page,
    Photo,
    Product,
    Review,
    Supplier,
    SupplierProfile,
    Topic,
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
def test_one_hop_lookup_path_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three products, each in its own category: one query per product would
    # show as more than one query.
    _create_product(name="Chair", category_name="Furniture")
    _create_product(name="Hammer", category_name="Tools")
    _create_product(name="Lamp", category_name="Lighting")
    rag.register(Product, fields=["name", "category__name"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nFurniture",
        "Hammer\n\nTools",
        "Lamp\n\nLighting",
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
def test_several_hop_lookup_path_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three reviews, each of a product in its own category: one query per
    # review or per product would show as more than one query.
    for title, product_name, category_name in [
        ("Sturdy", "Chair", "Furniture"),
        ("Heavy", "Hammer", "Tools"),
        ("Bright", "Lamp", "Lighting"),
    ]:
        product = _create_product(name=product_name, category_name=category_name)
        Review.objects.create(title=title, product=product)
    rag.register(Review, fields=["title", "product__category__name"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Sturdy\n\nFurniture",
        "Heavy\n\nTools",
        "Bright\n\nLighting",
    ]


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
def test_title_field_lookup_path_outside_fields_gives_the_title_not_text() -> None:
    product = _create_product(name="Chair", category_name="Furniture")
    Review.objects.create(title="Sturdy", product=product)
    rag.register(Review, fields=["title"], title_field="product__name")

    documents = SyncPipeline().run()

    assert [(document.title, document.text) for document in documents] == [
        ("Chair", "Sturdy")
    ]


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


@pytest.mark.django_db
def test_lookup_path_and_follow_through_the_same_relation_combine() -> None:
    topic = Topic.objects.create(
        title="Basics", summary="The very start", slug="basics"
    )
    Lesson.objects.create(title="Intro", topic=topic)
    rag.register(Lesson, fields=["title", "topic__slug"], follow=["topic"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Intro\n\nbasics\n\nBasics\n\nThe very start"
    ]


@pytest.mark.django_db
def test_lookup_path_and_follow_on_its_first_relation_are_read_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three reviews, each of a product in its own category: one query per
    # review or per product would show as more than one query.
    for title, product_name, category_name in [
        ("Sturdy", "Chair", "Furniture"),
        ("Heavy", "Hammer", "Tools"),
        ("Bright", "Lamp", "Lighting"),
    ]:
        product = _create_product(name=product_name, category_name=category_name)
        Review.objects.create(title=title, product=product)
    rag.register(
        Review, fields=["title", "product__category__name"], follow=["product"]
    )

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Sturdy\n\nFurniture\n\nChair\n\nNew",
        "Heavy\n\nTools\n\nHammer\n\nNew",
        "Bright\n\nLighting\n\nLamp\n\nNew",
    ]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("categroy__name", id="unknown-first-link"),
        pytest.param("category__nmae", id="unknown-last-link"),
    ],
)
def test_lookup_path_with_an_unknown_link_fails_at_registration_naming_the_path(
    path: str,
) -> None:
    with pytest.raises(ImproperlyConfigured, match=re.escape(path)):
        rag.register(Product, fields=["name", path])


def test_lookup_path_through_a_non_relation_fails_at_registration() -> None:
    # description is a text field of Product: it leads to no related model.
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Product, fields=["name", "description__name"])

    message = str(excinfo.value)
    assert "Product" in message
    assert "description__name" in message
    assert "not a relation" in message


def test_lookup_path_ending_on_a_relation_fails_at_registration() -> None:
    # category is a foreign key of Product: the path ends on a relation, not text.
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Review, fields=["title", "product__category"])

    message = str(excinfo.value)
    assert "product__category" in message
    assert "a relation" in message


@pytest.mark.parametrize(
    ("model", "fields", "path"),
    [
        pytest.param(
            Category, ["name", "products__name"], "products__name", id="reverse-fk"
        ),
        pytest.param(
            Course, ["title", "topics__title"], "topics__title", id="many-to-many"
        ),
        pytest.param(
            Photo, ["title", "tags__label"], "tags__label", id="generic-relation"
        ),
    ],
)
def test_lookup_path_through_a_many_valued_relation_fails_at_registration(
    model: type[Model], fields: list[str], path: str
) -> None:
    # Each first link can hold several objects: the path has no single value.
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(model, fields=fields)

    message = str(excinfo.value)
    assert path in message
    assert "several" in message
