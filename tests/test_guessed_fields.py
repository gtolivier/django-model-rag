import pytest
from django.db.models import Model

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    AccordionItem,
    Article,
    Category,
    EmailHolder,
    Note,
    Page,
    Panel,
    Product,
    SlugHolder,
    TextPlugin,
    URLHolder,
)


@pytest.mark.django_db
def test_model_registered_without_fields_produces_its_only_text_field() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Tools"]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_its_foreign_key() -> None:
    page = Page.objects.create(title="About us", slug="about")
    TextPlugin.objects.create(page=page, body="We build tools.")
    rag.register(TextPlugin)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["We build tools."]


@pytest.mark.django_db
def test_model_registered_without_fields_joins_its_text_fields_in_order() -> None:
    page = Page.objects.create(title="About us", slug="about")
    AccordionItem.objects.create(
        page=page, title="Opening hours", body="Monday to Friday."
    )
    rag.register(AccordionItem)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Opening hours\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_puts_its_title_first() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Opening hours\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_puts_its_title_like_fields_first() -> None:
    Panel.objects.create(
        body="Monday to Friday.",
        label="Hours",
        heading="When we are open",
        name="opening-hours",
        title="Opening hours",
    )
    rag.register(Panel)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Opening hours\n\nopening-hours\n\nWhen we are open\n\nHours"
        "\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_guesses_only_its_text_fields() -> None:
    category = Category.objects.create(name="Tools")
    Product.objects.create(
        name="Hammer",
        description="Drives nails.",
        subtitle=None,
        price="9.90",
        category=category,
        condition="used",
    )
    rag.register(Product)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Hammer\n\nDrives nails.\n\nSecond-hand"
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_guesses_a_text_field_subclass() -> None:
    Article.objects.create(body="Rich text, as a third-party field stores it.")
    rag.register(Article)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == [
        "Rich text, as a third-party field stores it."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_its_slug() -> None:
    Page.objects.create(title="About us", slug="about")
    rag.register(Page)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["About us"]


@pytest.mark.parametrize(
    ("model", "value"),
    [
        pytest.param(EmailHolder, "team@example.com", id="EmailField"),
        pytest.param(URLHolder, "https://example.com/about/", id="URLField"),
        pytest.param(SlugHolder, "about-us", id="SlugField"),
    ],
)
@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_a_char_field_subclass(
    model: type[Model], value: str
) -> None:
    model(label="Opening hours", extra=value).save()
    rag.register(model)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Opening hours"]
