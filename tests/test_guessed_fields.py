import pytest

from django_model_rag import SyncPipeline, rag
from tests.testapp.models import Category


@pytest.mark.django_db
def test_model_registered_without_fields_produces_its_only_text_field() -> None:
    Category.objects.create(name="Tools")
    rag.register(Category)

    documents = SyncPipeline().run()

    assert [document.text for document in documents] == ["Tools"]
