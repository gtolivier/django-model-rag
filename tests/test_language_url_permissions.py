import re

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.urls import NoReverseMatch
from pytest_django import DjangoAssertNumQueries

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    Announcement,
    Bookmark,
    Brochure,
    Bulletin,
    Category,
    Circular,
    Dispatch,
    Excerpt,
    Flyer,
    Gazette,
    Language,
    Leaflet,
    Memo,
    Notice,
    Page,
)


@pytest.mark.django_db
def test_model_without_language_field_produces_documents_without_language() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.language for document in documents] == [None]


@pytest.mark.django_db
def test_language_field_gives_each_document_its_instance_language() -> None:
    Bulletin.objects.create(title="Bonjour", locale="fr")
    Bulletin.objects.create(title="Hello", locale="en")
    rag.register(Bulletin, fields=["title"], language_field="locale")

    documents = SyncPipeline().run()

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

    documents = SyncPipeline().run()

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

    documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Blank": None,
        "Null": None,
    }


@pytest.mark.django_db
def test_own_field_named_language_gives_the_language_without_language_field() -> None:
    Notice.objects.create(title="Bonjour", language="  fr  ")
    Notice.objects.create(title="Hello", language="en")
    rag.register(Notice, fields=["title"])

    documents = SyncPipeline().run()

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

    documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_own_field_named_lang_gives_the_language_without_language_field() -> None:
    Dispatch.objects.create(title="Bonjour", lang="  fr  ")
    Dispatch.objects.create(title="Hello", lang="en")
    rag.register(Dispatch, fields=["title"])

    documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
    }


@pytest.mark.django_db
def test_guessed_language_field_is_chosen_per_model_by_name_not_per_instance() -> None:
    Gazette.objects.create(title="Both", language="fr", language_code="de")
    Gazette.objects.create(title="Code only", language="", language_code="en")
    rag.register(Gazette, fields=["title"])

    documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Both": "fr",
        "Code only": None,
    }


@pytest.mark.django_db
def test_relation_named_language_is_not_guessed_as_the_language() -> None:
    french = Language.objects.create(code="fr")
    Announcement.objects.create(title="Bonjour", language=french)
    rag.register(Announcement, fields=["title"])

    documents = SyncPipeline().run()

    assert [document.language for document in documents] == [None]


@pytest.mark.django_db
def test_language_field_is_left_out_of_guessed_text_fields_not_declared_ones() -> None:
    Notice.objects.create(title="Bonjour", language="fr")
    Bulletin.objects.create(title="Hello", locale="en")
    Circular.objects.create(title="Hallo", language_code="de")
    rag.register(Notice)
    rag.register(Bulletin, language_field="locale")
    rag.register(Circular, fields=["title", "language_code"])

    documents = SyncPipeline().run()

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

    documents = SyncPipeline().run()

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

    with django_assert_num_queries(1):
        documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Bonjour": "fr",
        "Hello": "en",
        "Hallo": "de",
    }


@pytest.mark.django_db
def test_constant_language_is_every_document_language_and_frees_the_guessed_field() -> (
    None
):
    Notice.objects.create(title="Bonjour", language="fr")
    Notice.objects.create(title="Hello", language="en")
    rag.register(Notice, language="de")

    documents = SyncPipeline().run()

    assert {document.text: document.language for document in documents} == {
        "Bonjour\n\nfr": "de",
        "Hello\n\nen": "de",
    }


@pytest.mark.django_db
def test_model_without_get_absolute_url_or_url_field_gives_an_empty_url() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.url for document in documents] == [""]


@pytest.mark.django_db
def test_get_absolute_url_gives_each_document_its_instance_url_left_relative() -> None:
    Page.objects.create(title="About", slug="about")
    Page.objects.create(title="Contact", slug="contact")
    rag.register(Page, fields=["title"])

    documents = SyncPipeline().run()

    assert {document.text: document.url for document in documents} == {
        "About": "/pages/about/",
        "Contact": "/pages/contact/",
    }


@pytest.mark.django_db
def test_run_instance_gives_the_document_the_url_of_its_get_absolute_url() -> None:
    page = Page.objects.create(title="About", slug="about")
    rag.register(Page, fields=["title"])

    documents = SyncPipeline().run_instance(page)

    assert [document.url for document in documents] == ["/pages/about/"]


@pytest.mark.django_db
def test_exception_raised_by_get_absolute_url_propagates_unchanged() -> None:
    Brochure.objects.create(title="Spring catalogue")
    rag.register(Brochure, fields=["title"])

    with pytest.raises(NoReverseMatch, match="brochure-detail"):
        SyncPipeline().run()


@pytest.mark.django_db
def test_get_absolute_url_returning_none_gives_an_empty_url_not_none() -> None:
    Flyer.objects.create(title="Spring sale")
    rag.register(Flyer, fields=["title"])

    documents = SyncPipeline().run()

    assert [document.url for document in documents] == [""]


@pytest.mark.django_db
def test_url_field_gives_each_document_its_stripped_value_relative_or_absolute() -> (
    None
):
    Bookmark.objects.create(title="Docs", link="  /docs/a/  ")
    Bookmark.objects.create(title="Example", link="https://example.com/b")
    rag.register(Bookmark, fields=["title"], url_field="link")

    documents = SyncPipeline().run()

    assert {document.text: document.url for document in documents} == {
        "Docs": "/docs/a/",
        "Example": "https://example.com/b",
    }


def test_language_field_the_model_lacks_fails_at_registration_naming_it() -> None:
    with pytest.raises(ImproperlyConfigured, match="nonexistent"):
        rag.register(Notice, fields=["title"], language_field="nonexistent")

    with pytest.raises(ImproperlyConfigured, match="notice__nonexistent"):
        rag.register(Excerpt, fields=["title"], language_field="notice__nonexistent")


def test_language_field_naming_a_relation_fails_at_registration_naming_it() -> None:
    with pytest.raises(ImproperlyConfigured, match="notice"):
        rag.register(Excerpt, fields=["title"], language_field="notice")

    with pytest.raises(ImproperlyConfigured, match="language"):
        rag.register(Announcement, fields=["title"], language_field="language")


def test_constant_language_with_language_field_fails_at_registration_naming_both() -> (
    None
):
    with pytest.raises(ImproperlyConfigured) as excinfo:
        rag.register(Notice, fields=["title"], language="fr", language_field="language")

    message = str(excinfo.value)
    assert "language_field" in message
    # "language" alone, not as the start of "language_field"
    assert re.search(r"\blanguage\b", message)


def test_blank_or_non_string_constant_language_fails_at_registration() -> None:
    with pytest.raises(ImproperlyConfigured):
        rag.register(Notice, fields=["title"], language="   ")

    with pytest.raises(ImproperlyConfigured):
        # A number is the slip under test: the type checker rightly rejects it.
        rag.register(Notice, fields=["title"], language=42)  # type: ignore[arg-type]
