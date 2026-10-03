import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Model

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    AccordionItem,
    Article,
    Category,
    CodeHolder,
    ContactCard,
    EmailHolder,
    Note,
    Page,
    Panel,
    Product,
    SlugHolder,
    StockLevel,
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
def test_model_registered_without_fields_takes_its_title_as_document_title() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note)

    [document] = SyncPipeline().run()

    assert document.title == "Opening hours"


@pytest.mark.django_db
def test_title_field_without_fields_gives_the_title_not_the_guessed_text() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note, title_field="body")

    [document] = SyncPipeline().run()

    assert (document.title, document.text) == (
        "Monday to Friday.",
        "Opening hours\n\nMonday to Friday.",
    )


@pytest.mark.django_db
def test_title_field_without_fields_may_name_a_field_that_is_not_guessed() -> None:
    Page.objects.create(title="About us", slug="about")
    rag.register(Page, title_field="slug")

    [document] = SyncPipeline().run()

    assert (document.title, document.text) == ("about", "About us")


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
def test_excluded_field_is_left_out_of_the_guessed_text_and_title() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note, exclude=["title"])

    [document] = SyncPipeline().run()

    assert (document.title, document.text) == (
        "Monday to Friday.",
        "Monday to Friday.",
    )


def test_excluding_a_field_the_model_does_not_have_names_it_in_the_error() -> None:
    with pytest.raises(ImproperlyConfigured, match="titel"):
        rag.register(Note, exclude=["titel"])


def test_excluding_a_field_twice_names_it_in_the_error() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\btitle\b"):
        rag.register(Note, exclude=["title", "title"])


def test_excluding_a_single_field_name_instead_of_a_list_fails() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bexclude\b"):
        # A bare string is the slip under test: the type checker rightly
        # rejects it.
        rag.register(Note, exclude="title")  # type: ignore[arg-type]


def test_model_without_text_field_registered_without_fields_names_it_in_the_error() -> (
    None
):
    with pytest.raises(ImproperlyConfigured, match=r"\bStockLevel\b"):
        rag.register(StockLevel)


def test_model_without_text_field_registered_without_fields_says_none_to_guess() -> (
    None
):
    with pytest.raises(ImproperlyConfigured, match=r"\bStockLevel\b.*\btext field\b"):
        rag.register(StockLevel)


def test_model_with_only_char_field_subclasses_registered_without_fields_names_it() -> (
    None
):
    with pytest.raises(ImproperlyConfigured, match=r"\bContactCard\b"):
        rag.register(ContactCard)


def test_excluding_every_guessed_field_names_the_model_in_the_error() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bNote\b"):
        rag.register(Note, exclude=["body", "title"])


def test_excluding_every_guessed_field_says_no_text_field_is_left() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"\bNote\b.*\btext field\b"):
        rag.register(Note, exclude=["body", "title"])


def test_exclude_cannot_be_combined_with_declared_fields() -> None:
    with pytest.raises(ImproperlyConfigured, match=r"fields.*exclude|exclude.*fields"):
        rag.register(Note, fields=["body"], exclude=["title"])


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


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_a_project_char_subclass() -> (
    None
):
    CodeHolder.objects.create(label="Opening hours", extra="FR")
    rag.register(CodeHolder)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Opening hours"]
