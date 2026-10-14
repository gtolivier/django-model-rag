import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Model

from django_model_rag import rag
from tests.recording import run_documents
from tests.testapp.models import (
    AccordionItem,
    Article,
    Category,
    CodeHolder,
    ContactCard,
    Country,
    Digest,
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

    documents = run_documents()

    assert [document.text for document in documents] == ["Tools"]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_its_foreign_key() -> None:
    page = Page.objects.create(title="About us", slug="about")
    TextPlugin.objects.create(page=page, body="We build tools.")
    rag.register(TextPlugin)

    documents = run_documents()

    assert [document.text for document in documents] == ["We build tools."]


@pytest.mark.django_db
def test_model_registered_without_fields_joins_its_text_fields_in_order() -> None:
    page = Page.objects.create(title="About us", slug="about")
    AccordionItem.objects.create(
        page=page, title="Opening hours", body="Monday to Friday."
    )
    rag.register(AccordionItem)

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Opening hours\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_puts_its_title_first() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note)

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Opening hours\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_takes_its_title_as_document_title() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note)

    [document] = run_documents()

    assert document.title == "Opening hours"


@pytest.mark.django_db
def test_title_field_without_fields_gives_the_title_not_the_guessed_text() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note, title_field="body")

    [document] = run_documents()

    assert (document.title, document.text) == (
        "Monday to Friday.",
        "Opening hours\n\nMonday to Friday.",
    )


@pytest.mark.django_db
def test_title_field_without_fields_may_name_a_field_that_is_not_guessed() -> None:
    Page.objects.create(title="About us", slug="about")
    rag.register(Page, title_field="slug")

    [document] = run_documents()

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

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Opening hours\n\nopening-hours\n\nWhen we are open\n\nHours"
        "\n\nMonday to Friday."
    ]


@pytest.mark.django_db
def test_excluded_field_is_left_out_of_the_guessed_text_and_title() -> None:
    Note.objects.create(body="Monday to Friday.", title="Opening hours")
    rag.register(Note, exclude=["title"])

    [document] = run_documents()

    assert (document.title, document.text) == (
        "Monday to Friday.",
        "Monday to Friday.",
    )


@pytest.mark.parametrize(
    ("model", "options", "match"),
    [
        pytest.param(
            Note, {"exclude": ["titel"]}, "titel", id="excluding-an-unknown-field"
        ),
        pytest.param(
            Note,
            {"exclude": ["title", "title"]},
            r"\btitle\b",
            id="excluding-a-field-twice",
        ),
        pytest.param(
            Note, {"exclude": "title"}, r"\bexclude\b", id="excluding-a-single-name"
        ),
        pytest.param(
            StockLevel,
            {},
            r"\bStockLevel\b.*\btext field\b",
            id="no-text-field-to-guess",
        ),
        pytest.param(
            ContactCard, {}, r"\bContactCard\b", id="only-char-field-subclasses"
        ),
        pytest.param(
            Note,
            {"exclude": ["body", "title"]},
            r"\bNote\b.*\btext field\b",
            id="excluding-every-guessed-field",
        ),
        pytest.param(
            Note,
            {"fields": ["body"], "exclude": ["title"]},
            r"fields.*exclude|exclude.*fields",
            id="exclude-with-declared-fields",
        ),
        pytest.param(
            Note,
            {"fields": ["body"], "exclude": ["title"]},
            r"^Note: ",
            id="exclude-with-declared-fields-names-the-model",
        ),
    ],
)
def test_registering_without_text_to_guess_or_with_invalid_exclude_fails(
    model: type[Model], options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ImproperlyConfigured, match=match):
        rag.register(model, **options)


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

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Hammer\n\nDrives nails.\n\nSecond-hand"
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_guesses_a_text_field_subclass() -> None:
    Article.objects.create(body="Rich text, as a third-party field stores it.")
    rag.register(Article)

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Rich text, as a third-party field stores it."
    ]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_its_slug() -> None:
    Page.objects.create(title="About us", slug="about")
    rag.register(Page)

    documents = run_documents()

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

    documents = run_documents()

    assert [document.text for document in documents] == ["Opening hours"]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_a_project_char_subclass() -> (
    None
):
    CodeHolder.objects.create(label="Opening hours", extra="FR")
    rag.register(CodeHolder)

    documents = run_documents()

    assert [document.text for document in documents] == ["Opening hours"]


@pytest.mark.django_db
def test_model_registered_without_fields_does_not_guess_its_char_primary_key() -> None:
    Country.objects.create(code="FR", name="France")
    rag.register(Country)

    documents = run_documents()

    assert [document.text for document in documents] == ["France"]


@pytest.mark.django_db
def test_model_registered_without_fields_guesses_its_non_editable_char_field() -> None:
    Digest.objects.create(title="Weekly news", summary="Three releases shipped.")
    rag.register(Digest)

    documents = run_documents()

    assert [document.text for document in documents] == [
        "Weekly news\n\nThree releases shipped."
    ]


def test_model_registered_without_fields_in_its_models_module_loads(
    tmp_path: Path,
) -> None:
    # The registration must run while Django loads the apps, as it does from
    # a project's models.py: this process's apps are long loaded, so a fresh
    # interpreter loads a throwaway app instead.
    app = tmp_path / "noticeboard"
    app.mkdir()
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(
        textwrap.dedent(
            """\
            from django.db import models

            from django_model_rag import rag


            class Memo(models.Model):
                body = models.TextField()


            rag.register(Memo)
            """
        )
    )
    script = tmp_path / "load_apps.py"
    script.write_text(
        textwrap.dedent(
            """\
            import django
            from django.conf import settings

            settings.configure(
                INSTALLED_APPS=["noticeboard"],
                DEFAULT_AUTO_FIELD="django.db.models.AutoField",
            )
            django.setup()

            from django_model_rag import rag

            print(*(model.__name__ for model in rag.registered_models()))
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

    assert result.stdout == "Memo\n", result.stderr
