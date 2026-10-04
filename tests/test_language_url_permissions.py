import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import (
    Announcement,
    Bulletin,
    Category,
    Circular,
    Dispatch,
    Gazette,
    Language,
    Leaflet,
    Memo,
    Notice,
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
