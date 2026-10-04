import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Bulletin, Category, Leaflet, Memo


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
