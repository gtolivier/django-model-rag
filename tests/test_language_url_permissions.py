import re
from typing import Any

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Model
from django.urls import NoReverseMatch
from pytest_django import DjangoAssertNumQueries

from django_model_rag import rag
from tests.recording import PRUNE_KEYS_QUERY, run_documents, run_instance_documents
from tests.testapp.models import (
    Announcement,
    Bookmark,
    Brochure,
    Bulletin,
    Category,
    Circular,
    Citation,
    Clipping,
    Dispatch,
    Excerpt,
    Flyer,
    Gazette,
    Language,
    Leaflet,
    Memo,
    Notice,
    Page,
    Pamphlet,
    Shortcut,
)


@pytest.mark.django_db
def test_model_without_language_field_produces_documents_without_language() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = run_documents()

    assert [document.language for document in documents] == [None]


@pytest.mark.django_db
def test_language_field_gives_each_document_its_instance_language() -> None:
    Bulletin.objects.create(title="Bonjour", locale="fr")
    Bulletin.objects.create(title="Hello", locale="en")
    rag.register(Bulletin, fields=["title"], language_field="locale")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_language_is_the_stored_value_stripped_not_a_choice_label() -> None:
    Leaflet.objects.create(title="Bonjour", locale="fr")
    Bulletin.objects.create(title="Hello", locale="  en  ")
    rag.register(Leaflet, fields=["title"], language_field="locale")
    rag.register(Bulletin, fields=["title"], language_field="locale")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_blank_or_null_language_gives_no_language() -> None:
    Bulletin.objects.create(title="Blank", locale="   ")
    Memo.objects.create(title="Null", locale=None)
    rag.register(Bulletin, fields=["title"], language_field="locale")
    rag.register(Memo, fields=["title"], language_field="locale")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Blank": None,
        "Null": None,
    }


@pytest.mark.django_db
def test_own_field_named_language_gives_the_language_without_language_field() -> None:
    Notice.objects.create(title="Bonjour", language="  fr  ")
    Notice.objects.create(title="Hello", language="en")
    rag.register(Notice, fields=["title"])

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_own_field_named_language_code_gives_the_language_without_language_field() -> (
    None
):
    Circular.objects.create(title="Bonjour", language_code="  fr  ")
    Circular.objects.create(title="Hello", language_code="en")
    rag.register(Circular, fields=["title"])

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_own_field_named_lang_gives_the_language_without_language_field() -> None:
    Dispatch.objects.create(title="Bonjour", lang="  fr  ")
    Dispatch.objects.create(title="Hello", lang="en")
    rag.register(Dispatch, fields=["title"])

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_guessed_language_field_is_chosen_per_model_by_name_not_per_instance() -> None:
    Gazette.objects.create(title="Both", language="fr", language_code="de")
    Gazette.objects.create(title="Code only", language="", language_code="en")
    rag.register(Gazette, fields=["title"])

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Both": "fr",
        "Code only": None,
    }


@pytest.mark.django_db
def test_relation_named_language_is_not_guessed_as_the_language() -> None:
    french = Language.objects.create(code="fr")
    Announcement.objects.create(title="Bonjour", language=french)
    rag.register(Announcement, fields=["title"])

    documents = run_documents()

    assert [document.language for document in documents] == [None]


@pytest.mark.django_db
def test_language_field_is_left_out_of_guessed_text_fields_not_declared_ones() -> None:
    Notice.objects.create(title="Bonjour", language="fr")
    Bulletin.objects.create(title="Hello", locale="en")
    Circular.objects.create(title="Hallo", language_code="de")
    rag.register(Notice)
    rag.register(Bulletin, language_field="locale")
    rag.register(Circular, fields=["title", "language_code"])

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
        "Hallo\n\nde": "de",
    }


@pytest.mark.django_db
def test_language_field_lookup_path_reads_the_language_of_the_related_object() -> None:
    french = Notice.objects.create(title="Avis", language="  fr  ")
    english = Notice.objects.create(title="Notice", language="en")
    Excerpt.objects.create(title="Bonjour", notice=french)
    Excerpt.objects.create(title="Hello", notice=english)
    Excerpt.objects.create(title="Orphan", notice=None)
    rag.register(Excerpt, fields=["title"], language_field="notice__language")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
        "Orphan": None,
    }


@pytest.mark.django_db
def test_language_field_lookup_path_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three excerpts, each of its own notice: one query per excerpt would show
    # as more than one query.
    for title, language in [("Bonjour", "fr"), ("Hello", "en"), ("Hallo", "de")]:
        notice = Notice.objects.create(title=title, language=language)
        Excerpt.objects.create(title=title, notice=notice)
    rag.register(Excerpt, fields=["title"], language_field="notice__language")

    with django_assert_num_queries(1 + PRUNE_KEYS_QUERY):
        documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
        "Hallo": "de",
    }


@pytest.mark.django_db
def test_language_field_lookup_path_gives_the_stored_code_not_a_choice_label() -> None:
    french = Leaflet.objects.create(title="Dépliant", locale="fr")
    english = Leaflet.objects.create(title="Leaflet", locale="en")
    Clipping.objects.create(title="Bonjour", leaflet=french)
    Clipping.objects.create(title="Hello", leaflet=english)
    rag.register(Clipping, fields=["title"], language_field="leaflet__locale")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_constant_language_is_every_document_language_and_frees_the_guessed_field() -> (
    None
):
    Notice.objects.create(title="Bonjour", language="fr")
    Notice.objects.create(title="Hello", language="en")
    rag.register(Notice, language="de")

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Bonjour\n\nfr": "de",
        "Hello\n\nen": "de",
    }


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        pytest.param(" fr ", "fr", id="spaces"),
        pytest.param("en\n", "en", id="newline"),
    ],
)
@pytest.mark.django_db
def test_constant_language_is_stripped_for_every_document(
    language: str, expected: str
) -> None:
    Category.objects.create(name="Tools")
    Category.objects.create(name="Garden")
    rag.register(Category, fields=["name"], language=language)

    documents = run_documents()

    assert {document.text: document.language for document in documents} == {
        "Tools": expected,
        "Garden": expected,
    }


@pytest.mark.django_db
def test_model_without_get_absolute_url_or_url_field_gives_an_empty_url() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = run_documents()

    assert [document.url for document in documents] == [""]


@pytest.mark.django_db
def test_get_absolute_url_gives_each_document_its_instance_url_left_relative() -> None:
    Page.objects.create(title="About", slug="about")
    Page.objects.create(title="Contact", slug="contact")
    rag.register(Page, fields=["title"])

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "About": "/pages/about/",
        "Contact": "/pages/contact/",
    }


@pytest.mark.django_db
def test_get_absolute_url_reads_undeclared_own_columns_loaded_with_the_instances(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Two pages whose url is built from their slug, which is not declared: a
    # slug left out of the select but read anyway would show as one more query
    # per page.
    Page.objects.create(title="About", slug="about")
    Page.objects.create(title="Contact", slug="contact")
    rag.register(Page, fields=["title"])

    with django_assert_num_queries(1 + PRUNE_KEYS_QUERY):
        documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "About": "/pages/about/",
        "Contact": "/pages/contact/",
    }


@pytest.mark.django_db
def test_run_instance_gives_the_document_the_url_of_its_get_absolute_url() -> None:
    page = Page.objects.create(title="About", slug="about")
    rag.register(Page, fields=["title"])

    documents = run_instance_documents(page)

    assert [document.url for document in documents] == ["/pages/about/"]


@pytest.mark.django_db
def test_exception_raised_by_get_absolute_url_propagates_unchanged() -> None:
    Brochure.objects.create(title="Spring catalogue")
    rag.register(Brochure, fields=["title"])

    with pytest.raises(NoReverseMatch, match="brochure-detail"):
        run_documents()


@pytest.mark.django_db
def test_get_absolute_url_returning_none_gives_an_empty_url_not_none() -> None:
    Flyer.objects.create(title="Spring sale")
    rag.register(Flyer, fields=["title"])

    documents = run_documents()

    assert [document.url for document in documents] == [""]


@pytest.mark.django_db
def test_url_field_gives_each_document_its_stripped_value_relative_or_absolute() -> (
    None
):
    Bookmark.objects.create(title="Docs", link="  /docs/a/  ")
    Bookmark.objects.create(title="Example", link="https://example.com/b")
    rag.register(Bookmark, fields=["title"], url_field="link")

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "Docs": "/docs/a/",
        "Example": "https://example.com/b",
    }


@pytest.mark.django_db
def test_url_field_wins_over_get_absolute_url_even_when_blank_or_null() -> None:
    Pamphlet.objects.create(title="Filled", link="  https://example.com/p  ")
    Pamphlet.objects.create(title="Blank", link="   ")
    Pamphlet.objects.create(title="Null", link=None)
    rag.register(Pamphlet, fields=["title"], url_field="link")

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "Filled": "https://example.com/p",
        "Blank": "",
        "Null": "",
    }


@pytest.mark.django_db
def test_url_field_with_choices_gives_the_stored_value_not_the_label() -> None:
    Shortcut.objects.create(title="Docs", link="/docs/")
    Shortcut.objects.create(title="Questions", link="/faq/")
    rag.register(Shortcut, fields=["title"], url_field="link")

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "Docs": "/docs/",
        "Questions": "/faq/",
    }


@pytest.mark.django_db
def test_url_field_is_left_out_of_guessed_text_fields() -> None:
    Bookmark.objects.create(title="Docs", link="/docs/a/")
    rag.register(Bookmark, url_field="link")

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "Docs": "/docs/a/",
    }


@pytest.mark.django_db
def test_url_field_lookup_path_reads_the_url_of_the_related_object() -> None:
    bookmark = Bookmark.objects.create(title="Docs", link="  /docs/a/  ")
    Citation.objects.create(title="Linked", bookmark=bookmark)
    Citation.objects.create(title="Orphan", bookmark=None)
    rag.register(Citation, fields=["title"], url_field="bookmark__link")

    documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "Linked": "/docs/a/",
        "Orphan": "",
    }


@pytest.mark.django_db
def test_url_field_lookup_path_is_read_with_its_instances_in_a_single_query(
    django_assert_num_queries: DjangoAssertNumQueries,
) -> None:
    # Three citations, each of its own bookmark: one query per citation would
    # show as more than one query.
    for title, link in [("A", "/docs/a/"), ("B", "/docs/b/"), ("C", "/docs/c/")]:
        bookmark = Bookmark.objects.create(title=title, link=link)
        Citation.objects.create(title=title, bookmark=bookmark)
    rag.register(Citation, fields=["title"], url_field="bookmark__link")

    with django_assert_num_queries(1 + PRUNE_KEYS_QUERY):
        documents = run_documents()

    assert {document.text: document.url for document in documents} == {
        "A": "/docs/a/",
        "B": "/docs/b/",
        "C": "/docs/c/",
    }


@pytest.mark.django_db
def test_model_without_permissions_option_gives_empty_frozenset_permissions() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = run_documents()

    assert [document.permissions for document in documents] == [frozenset()]
    assert all(isinstance(document.permissions, frozenset) for document in documents)


@pytest.mark.django_db
def test_permissions_option_gives_every_document_those_permissions() -> None:
    Category.objects.create(name="Tools")
    Category.objects.create(name="Garden")
    rag.register(
        Category,
        fields=["name"],
        permissions=["testapp.view_category", "testapp.change_category"],
    )

    documents = run_documents()

    expected = frozenset({"testapp.view_category", "testapp.change_category"})
    assert {document.text: document.permissions for document in documents} == {
        "Tools": expected,
        "Garden": expected,
    }
    assert all(isinstance(document.permissions, frozenset) for document in documents)


# Every permission is checked, not only the first: each bad one comes after
# a well-formed one.
WELL_FORMED_PERMISSION = "testapp.view_category"


@pytest.mark.parametrize(
    ("model", "options", "match"),
    [
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": "testapp.view_category"},
            r"\bpermissions\b",
            id="permissions-as-a-bare-string",
        ),
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": {"testapp.view_category"}},
            r"\bpermissions\b",
            id="permissions-as-a-set",
        ),
        pytest.param(
            Category,
            {
                "fields": ["name"],
                "permissions": [WELL_FORMED_PERMISSION, "view_category"],
            },
            re.escape("view_category"),
            id="permission-without-a-dot",
        ),
        pytest.param(
            Category,
            {
                "fields": ["name"],
                "permissions": [WELL_FORMED_PERMISSION, ".view_category"],
            },
            re.escape(".view_category"),
            id="permission-with-an-empty-app-label",
        ),
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": [WELL_FORMED_PERMISSION, "testapp."]},
            re.escape("testapp."),
            id="permission-with-an-empty-codename",
        ),
        # Its text form, "1.5", looks like app_label.codename.
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": [WELL_FORMED_PERMISSION, 1.5]},
            r"\bpermissions\b",
            id="permission-as-a-float-looking-well-formed",
        ),
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": [WELL_FORMED_PERMISSION, None]},
            r"\bpermissions\b",
            id="permission-as-none",
        ),
        pytest.param(
            Category,
            {"fields": ["name"], "permissions": [WELL_FORMED_PERMISSION, 42]},
            r"\bpermissions\b",
            id="permission-as-an-int",
        ),
        pytest.param(
            Notice,
            {"fields": ["title"], "language_field": "nonexistent"},
            "nonexistent",
            id="language-field-the-model-lacks",
        ),
        pytest.param(
            Excerpt,
            {"fields": ["title"], "language_field": "notice__nonexistent"},
            "notice__nonexistent",
            id="language-field-path-the-related-model-lacks",
        ),
        pytest.param(
            Excerpt,
            {"fields": ["title"], "language_field": "notice"},
            "notice",
            id="language-field-naming-a-relation",
        ),
        pytest.param(
            Announcement,
            {"fields": ["title"], "language_field": "language"},
            "language",
            id="language-field-naming-a-foreign-key",
        ),
        pytest.param(
            Bookmark,
            {"fields": ["title"], "url_field": "nonexistent"},
            "nonexistent",
            id="url-field-the-model-lacks",
        ),
        # A method, even the one that gives the url without url_field, is no
        # field.
        pytest.param(
            Pamphlet,
            {"fields": ["title"], "url_field": "get_absolute_url"},
            "get_absolute_url",
            id="url-field-naming-a-method",
        ),
        pytest.param(
            Citation,
            {"fields": ["title"], "url_field": "bookmark"},
            "bookmark",
            id="url-field-naming-a-relation",
        ),
        pytest.param(
            Notice,
            {"fields": ["title"], "language": "   "},
            None,
            id="blank-constant-language",
        ),
        pytest.param(
            Notice,
            {"fields": ["title"], "language": 42},
            None,
            id="non-string-constant-language",
        ),
    ],
)
def test_invalid_permissions_language_or_url_option_fails_at_registration(
    model: type[Model], options: dict[str, Any], match: str | None
) -> None:
    with pytest.raises(ImproperlyConfigured, match=match):
        rag.register(model, **options)


def test_constant_language_with_language_field_fails_at_registration_naming_both() -> (
    None
):
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Notice, fields=["title"], language="fr", language_field="language")

    message = str(excinfo.value)
    assert "language_field" in message
    # "language" alone, not as the start of "language_field"
    assert re.search(r"\blanguage\b", message)
