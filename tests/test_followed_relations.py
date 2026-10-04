import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ImproperlyConfigured
from pytest_django import DjangoAssertNumQueries

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    AccordionItem,
    Band,
    Category,
    Course,
    Craftsman,
    Delivery,
    Exhibit,
    FeaturedProduct,
    Guild,
    Lesson,
    Musician,
    Note,
    Page,
    PageIntro,
    Photo,
    Product,
    Recipe,
    Remark,
    Review,
    Shelf,
    Showroom,
    Step,
    StockLevel,
    Supplier,
    SupplierProfile,
    Tag,
    TextPlugin,
    Topic,
    Warehouse,
    Workshop,
)


def _create_chair(category: Category) -> Product:
    """Create a new product named "Chair", described as "Adjustable.", in ``category``.

    Its own text fields give "Chair", "Adjustable." and the condition's label,
    "New", in that order.
    """
    return Product.objects.create(
        name="Chair",
        description="Adjustable.",
        price="49.90",
        category=category,
        condition="new",
    )


def _create_woodworking_topics() -> tuple[Topic, Topic, Topic]:
    """Create the Joinery, Turning and Carving topics, in that primary key order.

    Their texts are "Joinery" / "Joints and finishes.", "Turning" / "Bowls and
    spindles." and "Carving" / "Spoons and reliefs.".
    """
    joinery = Topic.objects.create(
        summary="Joints and finishes.", title="Joinery", slug="joinery"
    )
    turning = Topic.objects.create(
        summary="Bowls and spindles.", title="Turning", slug="turning"
    )
    carving = Topic.objects.create(
        summary="Spoons and reliefs.", title="Carving", slug="carving"
    )
    return joinery, turning, carving


@pytest.mark.django_db
def test_followed_foreign_key_appends_the_related_text_after_the_own_fields() -> None:
    category = Category.objects.create(name="Furniture")
    _create_chair(category)
    rag.register(Product, follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nFurniture"
    ]


@pytest.mark.django_db
def test_run_instance_appends_the_followed_related_text_after_the_own_fields() -> None:
    category = Category.objects.create(name="Furniture")
    product = _create_chair(category)
    rag.register(Product, follow=["category"])

    documents = SyncPipeline().run_instance(product)

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nFurniture"
    ]


@pytest.mark.django_db
def test_followed_foreign_key_appends_the_related_text_after_the_declared_fields() -> (
    None
):
    # The fields are declared in the reverse of their declaration order, and
    # the condition is left out: only the declared fields come first.
    category = Category.objects.create(name="Furniture")
    _create_chair(category)
    rag.register(Product, fields=["description", "name"], follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Adjustable.\n\nChair\n\nFurniture"
    ]


@pytest.mark.django_db
def test_exclude_leaves_the_same_named_field_of_the_followed_related_in() -> None:
    # Product and Category both have a "name" field: excluding it leaves out
    # the product's own name only, never the followed category's.
    category = Category.objects.create(name="Furniture")
    _create_chair(category)
    rag.register(Product, exclude=["name"], follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Adjustable.\n\nNew\n\nFurniture"
    ]


@pytest.mark.django_db
def test_exclude_of_every_own_text_field_registers_with_a_followed_relation() -> None:
    # Without follow, excluding every text field of the model is refused: here
    # the followed category still gives the document a text, and no own field
    # is left to give a title, so the title is what str() gives.
    category = Category.objects.create(name="Furniture")
    product = Product.objects.create(
        name="Chair",
        description="Adjustable.",
        subtitle="Oak",
        price="49.90",
        category=category,
        condition="new",
    )
    rag.register(
        Product,
        exclude=["name", "description", "subtitle", "condition"],
        follow=["category"],
    )

    [document] = SyncPipeline().run()

    assert (document.title, document.text) == (str(product), "Furniture")


@pytest.mark.django_db
def test_followed_foreign_key_appends_each_instance_its_own_related_text() -> None:
    furniture = Category.objects.create(name="Furniture")
    lighting = Category.objects.create(name="Lighting")
    _create_chair(furniture)
    Product.objects.create(
        name="Lamp",
        description="Dimmable.",
        price="19.90",
        category=lighting,
        condition="new",
    )
    rag.register(Product, follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nFurniture",
        "Lamp\n\nDimmable.\n\nNew\n\nLighting",
    ]


@pytest.mark.django_db
def test_followed_foreign_key_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three products in two categories: one query per product, or per
    # category, would show as more than one query.
    furniture = Category.objects.create(name="Furniture")
    lighting = Category.objects.create(name="Lighting")
    _create_chair(furniture)
    Product.objects.create(
        name="Lamp",
        description="Dimmable.",
        price="19.90",
        category=lighting,
        condition="new",
    )
    Product.objects.create(
        name="Table",
        description="Extendable.",
        price="199.00",
        category=furniture,
        condition="new",
    )
    rag.register(Product, follow=["category"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nFurniture",
        "Lamp\n\nDimmable.\n\nNew\n\nLighting",
        "Table\n\nExtendable.\n\nNew\n\nFurniture",
    ]


@pytest.mark.django_db
def test_followed_foreign_key_brings_the_guessed_text_fields_of_the_related() -> None:
    topic = Topic.objects.create(
        summary="Joints and finishes.", title="Woodworking", slug="woodworking"
    )
    Lesson.objects.create(title="Dovetails", topic=topic)
    rag.register(Lesson, follow=["topic"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Dovetails\n\nWoodworking\n\nJoints and finishes."
    ]


@pytest.mark.django_db
def test_followed_foreign_key_loads_only_the_text_columns_of_the_related(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two lessons, each with its own topic: the topic's slug is not text, so
    # it is never read, and a text column left out of the select but read
    # anyway would show as one more query per lesson.
    joinery = Topic.objects.create(
        summary="Joints and finishes.", title="Joinery", slug="joinery"
    )
    turning = Topic.objects.create(
        summary="Bowls and spindles.", title="Turning", slug="turning"
    )
    Lesson.objects.create(title="Dovetails", topic=joinery)
    Lesson.objects.create(title="Spindles", topic=turning)
    rag.register(Lesson, follow=["topic"])

    with django_assert_num_queries(1) as queries:
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Dovetails\n\nJoinery\n\nJoints and finishes.",
        "Spindles\n\nTurning\n\nBowls and spindles.",
    ]
    assert '"testapp_topic"."slug"' not in queries.captured_queries[0]["sql"]


@pytest.mark.django_db
def test_followed_foreign_key_brings_the_label_of_a_related_field_with_choices() -> (
    None
):
    category = Category.objects.create(name="Tools")
    product = Product.objects.create(
        name="Hammer",
        description="Drives nails.",
        price="12.00",
        category=category,
        condition="used",
    )
    Review.objects.create(title="Still solid", product=product)
    rag.register(Review, follow=["product"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Still solid\n\nHammer\n\nDrives nails.\n\nSecond-hand"
    ]


@pytest.mark.django_db
def test_model_without_text_field_registers_when_a_followed_relation_brings_text() -> (
    None
):
    # StockLevel has only a number, a date and a boolean: registered alone it
    # is refused, but the followed product gives its document a text.
    category = Category.objects.create(name="Furniture")
    product = _create_chair(category)
    StockLevel.objects.create(
        quantity=12, counted_on=date(2026, 3, 1), in_stock=True, product=product
    )
    rag.register(StockLevel, follow=["product"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Chair\n\nAdjustable.\n\nNew"]


@pytest.mark.django_db
def test_model_without_text_field_that_follows_a_relation_takes_str_as_title() -> None:
    # StockLevel has no own text field to give a title, and defines no
    # __str__: the title is whatever str() gives for the instance.
    category = Category.objects.create(name="Furniture")
    product = _create_chair(category)
    stock_level = StockLevel.objects.create(
        quantity=12, counted_on=date(2026, 3, 1), in_stock=True, product=product
    )
    rag.register(StockLevel, follow=["product"])

    documents = SyncPipeline().run()

    assert [document.title for document in documents] == [str(stock_level)]


@pytest.mark.django_db
def test_model_without_text_field_that_follows_a_relation_takes_its_str_as_title() -> (
    None
):
    # Delivery has no own text field, but defines a __str__ built from its
    # quantity and date: the title is what that __str__ gives.
    category = Category.objects.create(name="Furniture")
    product = _create_chair(category)
    Delivery.objects.create(quantity=12, delivered_on=date(2026, 3, 1), product=product)
    rag.register(Delivery, follow=["product"])

    documents = SyncPipeline().run()

    assert [document.title for document in documents] == ["12 delivered on 2026-03-01"]


@pytest.mark.django_db
def test_title_from_str_loads_the_own_columns_str_reads_with_the_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two deliveries, each with a __str__ built from its quantity and date: a
    # column __str__ reads left out of the select would show as one more query
    # per delivery.
    category = Category.objects.create(name="Furniture")
    product = _create_chair(category)
    Delivery.objects.create(quantity=12, delivered_on=date(2026, 3, 1), product=product)
    Delivery.objects.create(quantity=5, delivered_on=date(2026, 4, 2), product=product)
    rag.register(Delivery, follow=["product"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.title for document in documents] == [
        "12 delivered on 2026-03-01",
        "5 delivered on 2026-04-02",
    ]


@pytest.mark.django_db
def test_followed_foreign_key_that_is_null_adds_nothing_to_the_own_fields() -> None:
    Workshop.objects.create(title="Open bench", topic=None)
    rag.register(Workshop, follow=["topic"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Open bench"]


@pytest.mark.django_db
def test_followed_foreign_key_whose_text_fields_are_blank_adds_nothing() -> None:
    topic = Topic.objects.create(summary="", title="   ", slug="blank")
    Lesson.objects.create(title="Dovetails", topic=topic)
    rag.register(Lesson, follow=["topic"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Dovetails"]


@pytest.mark.django_db
def test_instance_with_blank_own_text_has_a_document_when_a_followed_one_has_text() -> (
    None
):
    # Without follow, a lesson whose only text field is blank has no document:
    # here the followed topic gives it a text, and the blank title stays the
    # title, as its own fields give it.
    topic = Topic.objects.create(
        summary="Joints and finishes.", title="Woodworking", slug="woodworking"
    )
    Lesson.objects.create(title="   ", topic=topic)
    rag.register(Lesson, follow=["topic"])

    documents = SyncPipeline().run()

    assert [(document.title, document.text) for document in documents] == [
        ("", "Woodworking\n\nJoints and finishes.")
    ]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_appends_the_related_text() -> None:
    page = Page.objects.create(title="About us", slug="about-us")
    TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    rag.register(Page, follow=["text_plugins"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nWe build chairs by hand."
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_reverse_foreign_key_appends_the_related_texts_in_pk_order() -> None:
    page = Page.objects.create(title="About us", slug="about-us")
    TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    TextPlugin.objects.create(page=page, body="Our workshop is in Lyon.")
    TextPlugin.objects.create(page=page, body="Visits on Saturdays.")
    rag.register(Page, follow=["text_plugins"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nWe build chairs by hand.\n\nOur workshop is in Lyon."
        "\n\nVisits on Saturdays."
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_reverse_foreign_key_is_read_in_one_query_for_all_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three pages with two text plugins each: one query per page would show as
    # more than two queries, and the reversed selects check that each page's
    # texts still come in primary key order.
    about = Page.objects.create(title="About us", slug="about-us")
    contact = Page.objects.create(title="Contact", slug="contact")
    visits = Page.objects.create(title="Visits", slug="visits")
    TextPlugin.objects.create(page=about, body="We build chairs by hand.")
    TextPlugin.objects.create(page=contact, body="Write to us.")
    TextPlugin.objects.create(page=visits, body="Visits on Saturdays.")
    TextPlugin.objects.create(page=about, body="Our workshop is in Lyon.")
    TextPlugin.objects.create(page=contact, body="Or call us.")
    TextPlugin.objects.create(page=visits, body="Book a week ahead.")
    rag.register(Page, follow=["text_plugins"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nWe build chairs by hand.\n\nOur workshop is in Lyon.",
        "Contact\n\nWrite to us.\n\nOr call us.",
        "Visits\n\nVisits on Saturdays.\n\nBook a week ahead.",
    ]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_loads_only_the_text_columns_of_the_related(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two categories with two products each: the product's price is not text,
    # so it is never read, and a text column left out of the prefetch but read
    # anyway would show as one more query per product.
    furniture = Category.objects.create(name="Furniture")
    lighting = Category.objects.create(name="Lighting")
    _create_chair(furniture)
    Product.objects.create(
        name="Lamp",
        description="Dimmable.",
        price="19.90",
        category=lighting,
        condition="new",
    )
    Product.objects.create(
        name="Table",
        description="Extendable.",
        price="199.00",
        category=furniture,
        condition="new",
    )
    Product.objects.create(
        name="Bulb",
        description="Warm white.",
        price="4.50",
        category=lighting,
        condition="used",
    )
    rag.register(Category, follow=["products"])

    with django_assert_num_queries(2) as queries:
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Furniture\n\nChair\n\nAdjustable.\n\nNew\n\nTable\n\nExtendable.\n\nNew",
        "Lighting\n\nLamp\n\nDimmable.\n\nNew\n\nBulb\n\nWarm white.\n\nSecond-hand",
    ]
    assert '"testapp_product"."price"' not in queries.captured_queries[1]["sql"]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_to_a_unique_column_keeps_that_column_loaded(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two warehouses with two shelves each: Shelf.warehouse points to the
    # warehouse's code, not its primary key, so the prefetch matches the
    # shelves to their warehouse by that code. It is not declared, yet left
    # out of the select it would show as one more query per warehouse.
    north = Warehouse.objects.create(name="North depot", code="north")
    south = Warehouse.objects.create(name="South depot", code="south")
    Shelf.objects.create(label="Timber", warehouse=north)
    Shelf.objects.create(label="Glue", warehouse=south)
    Shelf.objects.create(label="Screws", warehouse=north)
    Shelf.objects.create(label="Varnish", warehouse=south)
    rag.register(Warehouse, fields=["name"], follow=["shelves"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "North depot\n\nTimber\n\nScrews",
        "South depot\n\nGlue\n\nVarnish",
    ]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_whose_manager_joins_another_foreign_key(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Exhibit's default manager joins its category with select_related(), and
    # the prefetch of the exhibits is built from it: deferring that foreign
    # key, though it is not text, would make Django refuse the query, and a
    # column the documents read left out would show as more than two queries.
    tools = Category.objects.create(name="Tools")
    north = Showroom.objects.create(name="North hall")
    south = Showroom.objects.create(name="South hall")
    Exhibit.objects.create(showroom=north, label="Hammer", category=tools)
    Exhibit.objects.create(showroom=south, label="Rake", category=tools)
    Exhibit.objects.create(showroom=north, label="Saw", category=tools)
    Exhibit.objects.create(showroom=south, label="Spade", category=tools)
    rag.register(Showroom, follow=["exhibits"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "North hall\n\nHammer\n\nSaw",
        "South hall\n\nRake\n\nSpade",
    ]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_without_related_objects_adds_nothing() -> None:
    Page.objects.create(title="About us", slug="about-us")
    rag.register(Page, follow=["text_plugins"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["About us"]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_reverse_foreign_key_without_related_name_uses_its_accessor() -> None:
    # Remark.note has no related_name: the relation is followed by its default
    # accessor, the name prefetch_related takes, not by its query name.
    note = Note.objects.create(title="Workshop rules", body="Wear goggles.")
    Remark.objects.create(note=note, body="Gloves too.")
    Remark.objects.create(note=note, body="Sweep the floor after use.")
    rag.register(Note, follow=["remark_set"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Workshop rules\n\nWear goggles.\n\nGloves too.\n\nSweep the floor after use."
    ]


@pytest.mark.django_db
def test_followed_reverse_foreign_key_to_a_model_whose_manager_is_not_a_manager() -> (
    None
):
    # Step's default manager comes from BaseManager.from_queryset(): Django
    # builds the reverse relation's manager from that class, so it derives from
    # BaseManager, not from Manager, and its steps are still the related text.
    recipe = Recipe.objects.create(title="Wood glue")
    # save(), not Step.objects.create(): the type stubs give a manager built
    # from BaseManager none of QuerySet's own methods.
    Step(recipe=recipe, body="Warm the hide glue.").save()
    rag.register(Recipe, follow=["steps"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Wood glue\n\nWarm the hide glue."
    ]


@pytest.mark.django_db
def test_followed_reverse_one_to_one_without_related_object_adds_nothing() -> None:
    # Only the second page has an intro: the first one's reverse one-to-one
    # has no object, where Django's accessor raises instead of giving None.
    Page.objects.create(title="About us", slug="about-us")
    contact = Page.objects.create(title="Contact", slug="contact")
    PageIntro.objects.create(page=contact, body="Write to us.")
    rag.register(Page, follow=["intro"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us",
        "Contact\n\nWrite to us.",
    ]


@pytest.mark.django_db
def test_followed_reverse_one_to_one_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three pages, the middle one without an intro: one query per page would
    # show as more than one query, and the page without an intro keeps only
    # its own text.
    about = Page.objects.create(title="About us", slug="about-us")
    Page.objects.create(title="Contact", slug="contact")
    visits = Page.objects.create(title="Visits", slug="visits")
    PageIntro.objects.create(page=about, body="We build chairs by hand.")
    PageIntro.objects.create(page=visits, body="Visits on Saturdays.")
    rag.register(Page, follow=["intro"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nWe build chairs by hand.",
        "Contact",
        "Visits\n\nVisits on Saturdays.",
    ]


@pytest.mark.django_db
def test_followed_reverse_one_to_one_with_its_own_query_name_uses_its_accessor(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # SupplierProfile.supplier sets a related_query_name, "supplier_profile",
    # that differs from its accessor, "profile": the relation is followed by
    # its accessor, and still read with the suppliers in a single query. Only
    # the second supplier has a profile.
    Supplier.objects.create(name="Oak & Co")
    birch = Supplier.objects.create(name="Birch Mill")
    SupplierProfile.objects.create(supplier=birch, body="Kiln-dried boards.")
    rag.register(Supplier, follow=["profile"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Oak & Co",
        "Birch Mill\n\nKiln-dried boards.",
    ]


@pytest.mark.django_db
def test_followed_forward_one_to_one_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three intros, each on its own page: one query per intro, or per page,
    # would show as more than one query.
    about = Page.objects.create(title="About us", slug="about-us")
    contact = Page.objects.create(title="Contact", slug="contact")
    visits = Page.objects.create(title="Visits", slug="visits")
    PageIntro.objects.create(page=about, body="We build chairs by hand.")
    PageIntro.objects.create(page=contact, body="Write to us.")
    PageIntro.objects.create(page=visits, body="Visits on Saturdays.")
    rag.register(PageIntro, follow=["page"])

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "We build chairs by hand.\n\nAbout us",
        "Write to us.\n\nContact",
        "Visits on Saturdays.\n\nVisits",
    ]


@pytest.mark.django_db
def test_child_model_follows_a_relation_inherited_from_its_parent() -> None:
    # FeaturedProduct inherits from Product through multi-table inheritance:
    # its automatic link to the parent row, product_ptr, is a one-to-one
    # relation too, and must not keep the inherited category from being
    # followed. The tagline, its own field, comes after the inherited ones.
    category = Category.objects.create(name="Furniture")
    FeaturedProduct.objects.create(
        name="Chair",
        description="Adjustable.",
        price="49.90",
        category=category,
        condition="new",
        tagline="Made in Lyon.",
    )
    rag.register(FeaturedProduct, follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nMade in Lyon.\n\nFurniture"
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_many_to_many_appends_the_related_texts_in_pk_order() -> None:
    # The topics are linked in an order other than their primary keys', so
    # neither the link order nor the reversed selects can give pk order.
    joinery, turning, carving = _create_woodworking_topics()
    course = Course.objects.create(title="Woodworking basics")
    course.topics.add(turning, carving, joinery)
    rag.register(Course, follow=["topics"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Woodworking basics\n\nJoinery\n\nJoints and finishes."
        "\n\nTurning\n\nBowls and spindles.\n\nCarving\n\nSpoons and reliefs."
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_many_to_many_is_read_in_one_query_for_all_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three courses with two topics each, some topics shared: one query per
    # course would show as more than two queries, and the topics are linked in
    # an order other than their primary keys', so neither the link order nor
    # the reversed selects can give each course its texts in pk order.
    joinery, turning, carving = _create_woodworking_topics()
    basics = Course.objects.create(title="Woodworking basics")
    furniture = Course.objects.create(title="Furniture making")
    restoration = Course.objects.create(title="Antique restoration")
    basics.topics.add(turning, joinery)
    furniture.topics.add(carving, joinery)
    restoration.topics.add(carving, turning)
    rag.register(Course, follow=["topics"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Woodworking basics\n\nJoinery\n\nJoints and finishes."
        "\n\nTurning\n\nBowls and spindles.",
        "Furniture making\n\nJoinery\n\nJoints and finishes."
        "\n\nCarving\n\nSpoons and reliefs.",
        "Antique restoration\n\nTurning\n\nBowls and spindles."
        "\n\nCarving\n\nSpoons and reliefs.",
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_many_to_many_loads_only_the_text_columns_of_the_related(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two courses with two topics each: the topic's slug is not text, so it is
    # never read, and a text column left out of the prefetch but read anyway
    # would show as one more query per topic.
    joinery, turning, carving = _create_woodworking_topics()
    basics = Course.objects.create(title="Woodworking basics")
    furniture = Course.objects.create(title="Furniture making")
    basics.topics.add(turning, joinery)
    furniture.topics.add(carving, joinery)
    rag.register(Course, follow=["topics"])

    with django_assert_num_queries(2) as queries:
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Woodworking basics\n\nJoinery\n\nJoints and finishes."
        "\n\nTurning\n\nBowls and spindles.",
        "Furniture making\n\nJoinery\n\nJoints and finishes."
        "\n\nCarving\n\nSpoons and reliefs.",
    ]
    assert '"testapp_topic"."slug"' not in queries.captured_queries[1]["sql"]


@pytest.mark.django_db
def test_followed_many_to_many_through_a_key_to_a_unique_column_keeps_it_loaded(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two guilds with two members each: Membership.guild points to the guild's
    # code, not its primary key, so the prefetch matches the members to their
    # guild by that code. It is not declared, yet left out of the select it
    # would show as one more query per guild.
    carpenter = Craftsman.objects.create(name="Carpenter")
    smith = Craftsman.objects.create(name="Smith")
    weaver = Craftsman.objects.create(name="Weaver")
    north = Guild.objects.create(name="North guild", code="north")
    south = Guild.objects.create(name="South guild", code="south")
    north.members.add(carpenter, weaver)
    south.members.add(smith, weaver)
    rag.register(Guild, fields=["name"], follow=["members"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "North guild\n\nCarpenter\n\nWeaver",
        "South guild\n\nSmith\n\nWeaver",
    ]


@pytest.mark.django_db
def test_followed_reverse_many_to_many_through_a_key_to_a_unique_column_keeps_it(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two musicians in two bands each: Engagement.musician points to the
    # musician's handle, not its primary key, so the prefetch matches the bands
    # to their musician by that handle. It is not declared, yet left out of the
    # select it would show as one more query per musician.
    quartet = Band.objects.create(name="Quartet")
    trio = Band.objects.create(name="Trio")
    duo = Band.objects.create(name="Duo")
    ada = Musician.objects.create(name="Ada", handle="ada")
    ben = Musician.objects.create(name="Ben", handle="ben")
    ada.bands.add(quartet, duo)
    ben.bands.add(trio, duo)
    rag.register(Musician, fields=["name"], follow=["bands"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Ada\n\nQuartet\n\nDuo",
        "Ben\n\nTrio\n\nDuo",
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_reverse_many_to_many_appends_the_related_texts_in_pk_order() -> None:
    # The courses are linked in an order other than their primary keys', so
    # neither the link order nor the reversed selects can give pk order.
    topic = Topic.objects.create(
        summary="Joints and finishes.", title="Joinery", slug="joinery"
    )
    basics = Course.objects.create(title="Woodworking basics")
    furniture = Course.objects.create(title="Furniture making")
    restoration = Course.objects.create(title="Antique restoration")
    topic.courses.add(furniture, restoration, basics)
    rag.register(Topic, follow=["courses"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Joinery\n\nJoints and finishes.\n\nWoodworking basics"
        "\n\nFurniture making\n\nAntique restoration"
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_reverse_many_to_many_is_read_in_one_query_for_all_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three topics in two courses each, some courses shared: one query per
    # topic would show as more than two queries, and the courses are linked in
    # an order other than their primary keys', so neither the link order nor
    # the reversed selects can give each topic its texts in pk order.
    joinery, turning, carving = _create_woodworking_topics()
    basics = Course.objects.create(title="Woodworking basics")
    furniture = Course.objects.create(title="Furniture making")
    restoration = Course.objects.create(title="Antique restoration")
    joinery.courses.add(furniture, basics)
    turning.courses.add(restoration, basics)
    carving.courses.add(restoration, furniture)
    rag.register(Topic, follow=["courses"])

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Joinery\n\nJoints and finishes.\n\nWoodworking basics\n\nFurniture making",
        "Turning\n\nBowls and spindles.\n\nWoodworking basics\n\nAntique restoration",
        "Carving\n\nSpoons and reliefs.\n\nFurniture making\n\nAntique restoration",
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_generic_relation_is_read_in_one_query_for_all_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three photos with two tags each: one query per photo would show as more
    # than two queries, and the reversed selects check that each photo's
    # texts still come in primary key order.
    workbench = Photo.objects.create(title="Workbench")
    lathe = Photo.objects.create(title="Lathe")
    chisels = Photo.objects.create(title="Chisels")
    Tag.objects.create(content_object=workbench, label="Oak")
    Tag.objects.create(content_object=lathe, label="Turning")
    Tag.objects.create(content_object=chisels, label="Carving")
    Tag.objects.create(content_object=workbench, label="Joinery")
    Tag.objects.create(content_object=lathe, label="Bowls")
    Tag.objects.create(content_object=chisels, label="Sharpening")
    rag.register(Photo, follow=["tags"])
    # Django caches content types per process: whether Photo's is already
    # cached depends on the tests run before, and a cold cache would add a
    # query to the count. Clearing then warming it makes the count the same
    # whatever the test order.
    ContentType.objects.clear_cache()
    ContentType.objects.get_for_model(Photo)

    with django_assert_num_queries(2):
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Workbench\n\nOak\n\nJoinery",
        "Lathe\n\nTurning\n\nBowls",
        "Chisels\n\nCarving\n\nSharpening",
    ]


@pytest.mark.django_db
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_generic_relation_loads_only_the_text_columns_of_the_related(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two photos with two tags each: the tag's weight is not text, so it is
    # never read, and a text column or a column of the link back left out of
    # the prefetch but read anyway would show as one more query per tag.
    workbench = Photo.objects.create(title="Workbench")
    lathe = Photo.objects.create(title="Lathe")
    Tag.objects.create(content_object=workbench, label="Oak", weight=3)
    Tag.objects.create(content_object=lathe, label="Turning", weight=5)
    Tag.objects.create(content_object=workbench, label="Joinery", weight=1)
    Tag.objects.create(content_object=lathe, label="Bowls", weight=2)
    rag.register(Photo, follow=["tags"])
    # Django caches content types per process: clearing then warming the
    # cache keeps a cold one from adding a query to the count.
    ContentType.objects.clear_cache()
    ContentType.objects.get_for_model(Photo)

    with django_assert_num_queries(2) as queries:
        documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Workbench\n\nOak\n\nJoinery",
        "Lathe\n\nTurning\n\nBowls",
    ]
    assert '"testapp_tag"."weight"' not in queries.captured_queries[1]["sql"]


@pytest.mark.django_db
def test_followed_relations_append_their_texts_in_the_order_follow_names_them() -> None:
    # TextPlugin is declared before AccordionItem: the order of ``follow``
    # differs from the declaration order of the related models.
    page = Page.objects.create(title="About us", slug="about-us")
    TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    AccordionItem.objects.create(
        page=page, title="Opening hours", body="Visits on Saturdays."
    )
    rag.register(Page, follow=["accordion_items", "text_plugins"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nOpening hours\n\nVisits on Saturdays.\n\nWe build chairs by hand."
    ]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("categroy", id="unknown-name"),
        # Only one level is followed: a lookup path is not a relation accessor.
        pytest.param("category__name", id="lookup-path"),
    ],
)
def test_following_a_name_that_is_not_a_relation_accessor_fails_at_registration(
    name: str,
) -> None:
    with pytest.raises(
        ImproperlyConfigured, match=rf"\b{name}\b.*\bfollow\b|\bfollow\b.*\b{name}\b"
    ):
        rag.register(Product, follow=[name])


def test_following_the_query_name_of_a_reverse_foreign_key_fails_at_registration() -> (
    None
):
    # Remark.note has no related_name: its query name "remark" differs from its
    # accessor "remark_set", and follow takes accessors, as prefetch_related does.
    with pytest.raises(
        ImproperlyConfigured, match=r"\bremark\b.*\bfollow\b|\bfollow\b.*\bremark\b"
    ):
        rag.register(Note, follow=["remark"])


def test_following_a_field_that_is_not_a_relation_fails_at_registration() -> None:
    # Product.name exists but is a CharField: the message must say why an
    # existing field is refused, not only that it cannot be followed.
    with pytest.raises(
        ImproperlyConfigured,
        match=r"\bname\b.*\bnot a relation\b|\bnot a relation\b.*\bname\b",
    ):
        rag.register(Product, follow=["name"])


def test_following_a_relation_twice_names_it_in_the_error() -> None:
    with pytest.raises(
        ImproperlyConfigured,
        match=r"\bcategory\b.*\btwice\b|\btwice\b.*\bcategory\b",
    ):
        rag.register(Product, follow=["category", "category"])


def test_following_a_relation_whose_model_has_no_text_field_fails() -> None:
    # StockLevel has only a number, a date and a boolean: following it could
    # never bring any text.
    with pytest.raises(
        ImproperlyConfigured, match=r"\bstock_levels\b.*\bno text field\b"
    ):
        rag.register(Product, follow=["stock_levels"])


def test_following_a_generic_foreign_key_fails_at_registration() -> None:
    # Tag.content_object may point to an instance of any model: its related
    # model is unknown, so no text could be guessed for it.
    with pytest.raises(
        ImproperlyConfigured,
        match=r"\bcontent_object\b.*\bgeneric foreign key\b"
        r"|\bgeneric foreign key\b.*\bcontent_object\b",
    ):
        rag.register(Tag, follow=["content_object"])


def test_following_a_single_relation_name_instead_of_a_list_fails() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bfollow\b.*\blist or a tuple\b"):
        # A bare string is the slip under test: the type checker rightly
        # rejects it.
        rag.register(Product, follow="category")  # type: ignore[arg-type]


def test_following_none_on_a_model_without_text_field_fails_on_its_type() -> None:
    # StockLevel has no own text field: the type of follow must be refused
    # before any guess of the fields could fail on their absence.
    with pytest.raises(ImproperlyConfigured, match=r"\bfollow\b.*\blist or a tuple\b"):
        # None is the slip under test: the type checker rightly rejects it.
        rag.register(StockLevel, follow=None)  # type: ignore[arg-type]


def test_following_a_relation_from_a_models_module_fails_and_points_to_appconfig_ready(
    tmp_path: Path,
) -> None:
    # Following a relation needs every model loaded, which is not the case
    # while a models.py runs: this process's apps are long loaded, so a fresh
    # interpreter loads a throwaway app that registers from its models.py.
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


            rag.register(Memo, follow=["board"])
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
    assert "rag.py" in result.stdout, result.stdout
    assert "AppConfig.ready()" in result.stdout, result.stdout
