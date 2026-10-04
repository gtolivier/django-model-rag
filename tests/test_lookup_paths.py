import re
import subprocess
import sys
import textwrap
from pathlib import Path

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
    SupplierOrder,
    SupplierProfile,
    Tag,
    Topic,
    Workshop,
)


def _create_product(*, name: str, category_name: str) -> Product:
    """Create a product called ``name`` in a new category called ``category_name``."""
    category = Category.objects.create(name=category_name)
    return Product.objects.create(
        name=name, description="", price="9.90", category=category
    )


def _create_three_reviews_of_products_in_their_own_categories() -> None:
    """Create Sturdy/Chair/Furniture, Heavy/Hammer/Tools and Bright/Lamp/Lighting.

    One query per review or per product would show as more than one query.
    """
    for title, product_name, category_name in [
        ("Sturdy", "Chair", "Furniture"),
        ("Heavy", "Hammer", "Tools"),
        ("Bright", "Lamp", "Lighting"),
    ]:
        product = _create_product(name=product_name, category_name=category_name)
        Review.objects.create(title=title, product=product)


def _create_supplier(*, name: str, profile_body: str) -> Supplier:
    """Create a supplier called ``name``, with a profile of body ``profile_body``."""
    supplier = Supplier.objects.create(name=name)
    SupplierProfile.objects.create(supplier=supplier, body=profile_body)
    return supplier


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
    _create_three_reviews_of_products_in_their_own_categories()
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
def test_title_field_lookup_path_outside_fields_is_read_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    _create_three_reviews_of_products_in_their_own_categories()
    rag.register(Review, fields=["title"], title_field="product__category__name")

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [(document.title, document.text) for document in documents] == [
        ("Furniture", "Sturdy"),
        ("Tools", "Heavy"),
        ("Lighting", "Bright"),
    ]


@pytest.mark.django_db
def test_lookup_path_through_a_null_foreign_key_adds_nothing() -> None:
    Workshop.objects.create(title="Pottery", topic=None)
    rag.register(Workshop, fields=["title", "topic__title"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Pottery"]


@pytest.mark.django_db
def test_lookup_path_crossing_a_reverse_one_to_one_by_its_query_name() -> None:
    _create_supplier(name="Acme", profile_body="Fine tools since 1920")
    rag.register(Supplier, fields=["name", "supplier_profile__body"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Acme\n\nFine tools since 1920"
    ]


@pytest.mark.django_db
def test_reverse_one_to_one_lookup_path_by_query_name_is_read_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three suppliers, each with its own profile: one query per supplier would
    # show as more than one query. The path names the relation by its query
    # name (supplier_profile), which differs from its accessor (profile).
    for name, body in [
        ("Acme", "Fine tools since 1920"),
        ("Globex", "Lamps for every room"),
        ("Initech", "Chairs built to last"),
    ]:
        _create_supplier(name=name, profile_body=body)
    rag.register(Supplier, fields=["name", "supplier_profile__body"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Acme\n\nFine tools since 1920",
        "Globex\n\nLamps for every room",
        "Initech\n\nChairs built to last",
    ]


@pytest.mark.django_db
def test_reverse_one_to_one_past_the_first_link_by_query_name_is_read_in_one_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three orders, each from a supplier with its own profile: one query per
    # order or per supplier would show as more than one query. The path names
    # the reverse one-to-one, past its first link, by its query name
    # (supplier_profile), which differs from its accessor (profile).
    for reference, name, body in [
        ("PO-1", "Acme", "Fine tools since 1920"),
        ("PO-2", "Globex", "Lamps for every room"),
        ("PO-3", "Initech", "Chairs built to last"),
    ]:
        supplier = _create_supplier(name=name, profile_body=body)
        SupplierOrder.objects.create(reference=reference, supplier=supplier)
    rag.register(
        SupplierOrder, fields=["reference", "supplier__supplier_profile__body"]
    )

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "PO-1\n\nFine tools since 1920",
        "PO-2\n\nLamps for every room",
        "PO-3\n\nChairs built to last",
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
    _create_three_reviews_of_products_in_their_own_categories()
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


def test_lookup_path_with_an_unknown_last_link_names_the_registered_model() -> None:
    # category leads to Category, which has no field nmae: the error is about
    # the model being registered and its whole path, not about Category.
    with pytest.raises(
        ImproperlyConfigured,
        match=re.escape("Product has no field 'category__nmae'"),
    ):
        rag.register(Product, fields=["name", "category__nmae"])


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


@pytest.mark.parametrize(
    ("model", "fields", "path"),
    [
        pytest.param(Product, ["name"], "categroy__name", id="unknown-link"),
        pytest.param(Review, ["title"], "product__category", id="ending-on-relation"),
        pytest.param(Category, ["name"], "products__name", id="many-valued-link"),
    ],
)
def test_title_field_invalid_lookup_path_fails_at_registration_naming_the_path(
    model: type[Model], fields: list[str], path: str
) -> None:
    with pytest.raises(ImproperlyConfigured, match=re.escape(path)):
        rag.register(model, fields=fields, title_field=path)


def test_lookup_path_declared_twice_fails_at_registration_naming_it() -> None:
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Product, fields=["name", "category__name", "category__name"])

    message = str(excinfo.value)
    assert "category__name" in message
    assert "twice" in message


def test_lookup_path_through_a_generic_foreign_key_fails_at_registration() -> None:
    # Tag.content_object may point to an instance of any model: its related
    # model changes from one row to the next, so the path cannot be resolved.
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Tag, fields=["label", "content_object__name"])

    message = str(excinfo.value)
    assert "content_object__name" in message
    assert "generic foreign key" in message
    # A generic foreign key is a relation, only one without a single model.
    assert "not a relation" not in message


def test_lookup_path_in_exclude_fails_at_registration_naming_it() -> None:
    # exclude removes fields from the model's own guessed text fields: a path
    # through a relation is never among them, so there is nothing to remove.
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Product, exclude=["category__name"])

    message = str(excinfo.value)
    assert "category__name" in message
    assert "own field" in message


def _register_a_lookup_path_from_a_models_module(tmp_path: Path) -> str:
    """Return the ImproperlyConfigured message of a registration in a models.py.

    Resolving a lookup path needs the related models loaded, which is not the
    case while a models.py runs: this process's apps are long loaded, so a
    fresh interpreter loads a throwaway app that registers from its models.py.
    """
    app = tmp_path / "noticeboard"
    app.mkdir()
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(
        textwrap.dedent(
            """\
            from django.db import models

            from django_model_rag import rag


            class Board(models.Model):
                name = models.CharField(max_length=100)


            class Memo(models.Model):
                body = models.TextField()
                board = models.ForeignKey(Board, on_delete=models.CASCADE)


            rag.register(Memo, fields=["body", "board__name"])
            """
        )
    )
    script = tmp_path / "load_apps.py"
    script.write_text(
        textwrap.dedent(
            """\
            import django
            from django.conf import settings
            from django.core.exceptions import ImproperlyConfigured

            settings.configure(
                INSTALLED_APPS=["noticeboard"],
                DEFAULT_AUTO_FIELD="django.db.models.AutoField",
            )
            try:
                django.setup()
            except ImproperlyConfigured as error:
                print(error)
            """
        )
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.stdout, f"no ImproperlyConfigured raised; stderr:\n{result.stderr}"
    return result.stdout


def test_lookup_path_from_a_models_module_fails_and_points_to_appconfig_ready(
    tmp_path: Path,
) -> None:
    message = _register_a_lookup_path_from_a_models_module(tmp_path)

    assert "rag.py" in message, message
    assert "AppConfig.ready()" in message, message


def test_lookup_path_from_a_models_module_fails_naming_the_lookup_path(
    tmp_path: Path,
) -> None:
    message = _register_a_lookup_path_from_a_models_module(tmp_path)

    assert "lookup path" in message, message
    assert "'board__name'" in message, message
    # The registration uses fields only: it never asked to follow anything.
    assert "follow relations" not in message, message
