import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    Category,
    Lesson,
    Page,
    Product,
    Review,
    TextPlugin,
    Topic,
    Workshop,
)


@pytest.mark.django_db
def test_followed_foreign_key_appends_the_related_text_after_the_own_fields() -> None:
    category = Category.objects.create(name="Furniture")
    Product.objects.create(
        name="Chair",
        description="Adjustable.",
        price="49.90",
        category=category,
        condition="new",
    )
    rag.register(Product, follow=["category"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Chair\n\nAdjustable.\n\nNew\n\nFurniture"
    ]


@pytest.mark.django_db
def test_followed_foreign_key_appends_each_instance_its_own_related_text() -> None:
    furniture = Category.objects.create(name="Furniture")
    lighting = Category.objects.create(name="Lighting")
    Product.objects.create(
        name="Chair",
        description="Adjustable.",
        price="49.90",
        category=furniture,
        condition="new",
    )
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
def test_followed_reverse_foreign_key_appends_the_related_text() -> None:
    page = Page.objects.create(title="About us", slug="about-us")
    TextPlugin.objects.create(page=page, body="We build chairs by hand.")
    rag.register(Page, follow=["text_plugins"])

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "About us\n\nWe build chairs by hand."
    ]
