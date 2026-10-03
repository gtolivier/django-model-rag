import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Category, Page, TextPlugin


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
