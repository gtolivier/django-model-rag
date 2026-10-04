import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Category


@pytest.mark.django_db
def test_model_without_language_field_produces_documents_without_language() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category, fields=["name"])

    documents = SyncPipeline().run()

    assert [document.language for document in documents] == [None]
