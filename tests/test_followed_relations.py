import pytest
from django.core.exceptions import ImproperlyConfigured

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    AccordionItem,
    Category,
    Course,
    Lesson,
    Note,
    Page,
    PageIntro,
    Product,
    Remark,
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
@pytest.mark.usefixtures("unordered_selects_reversed")
def test_followed_many_to_many_appends_the_related_texts_in_pk_order() -> None:
    # The topics are linked in an order other than their primary keys', so
    # neither the link order nor the reversed selects can give pk order.
    joinery = Topic.objects.create(
        summary="Joints and finishes.", title="Joinery", slug="joinery"
    )
    turning = Topic.objects.create(
        summary="Bowls and spindles.", title="Turning", slug="turning"
    )
    carving = Topic.objects.create(
        summary="Spoons and reliefs.", title="Carving", slug="carving"
    )
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


def test_following_a_single_relation_name_instead_of_a_list_fails() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bfollow\b.*\blist or a tuple\b"):
        # A bare string is the slip under test: the type checker rightly
        # rejects it.
        rag.register(Product, follow="category")  # type: ignore[arg-type]
