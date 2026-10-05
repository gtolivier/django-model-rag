from collections.abc import Iterator, Mapping, Sequence

import pytest
from pytest_django import DjangoCaptureOnCommitCallbacks, Settings

from django_model_rag import BaseExtractor, NormalizedDocument, rag
from tests.recording import TrackedRecordingOutput
from tests.testapp.models import Category

# The dotted path of the backend whose built instances the tests read back.
TRACKED_BACKEND = "tests.recording.TrackedRecordingOutput"


@pytest.fixture
def built_outputs() -> Iterator[list[TrackedRecordingOutput]]:
    """The TrackedRecordingOutput instances built during the test, and only those."""
    TrackedRecordingOutput.built.clear()
    yield TrackedRecordingOutput.built
    TrackedRecordingOutput.built.clear()


def _replaced(
    built_outputs: list[TrackedRecordingOutput],
) -> list[Mapping[str, Sequence[NormalizedDocument]]]:
    """Every replace call received, across every output built, in call order."""
    return [groups for output in built_outputs for groups in output.replaced]


@pytest.mark.django_db
def test_saving_a_registered_instance_replaces_its_group_once_its_transaction_commits(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    django_capture_on_commit_callbacks: DjangoCaptureOnCommitCallbacks,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    @rag.register_extractor(Category)
    class CategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    with django_capture_on_commit_callbacks(execute=True):
        lighting = Category.objects.create(name="Lighting")
        # A rolled-back transaction must leave the output untouched: nothing
        # may reach it before the commit.
        assert _replaced(built_outputs) == []

    assert _replaced(built_outputs) == [
        {
            f"testapp.category:{lighting.pk}": [
                NormalizedDocument(
                    text="Lighting",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=lighting.pk,
                ),
            ],
        }
    ]
