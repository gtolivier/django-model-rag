from collections.abc import Iterator

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from pytest_django import Settings

from django_model_rag import rag
from tests.recording import PruneOnlyOutput, TrackedRecordingOutput
from tests.testapp.models import Category, Product


@pytest.fixture
def built_outputs() -> Iterator[list[TrackedRecordingOutput]]:
    """The TrackedRecordingOutput instances built during the test, and only those."""
    TrackedRecordingOutput.built.clear()
    yield TrackedRecordingOutput.built
    TrackedRecordingOutput.built.clear()


@pytest.fixture
def built_prune_only_outputs() -> Iterator[list[PruneOnlyOutput]]:
    """The PruneOnlyOutput instances built during the test, and only those."""
    PruneOnlyOutput.built.clear()
    yield PruneOnlyOutput.built
    PruneOnlyOutput.built.clear()


def test_the_command_without_an_output_setting_names_the_missing_setting() -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        call_command("sync_model_rag")


def test_the_command_with_an_output_setting_that_is_not_a_dict_names_the_setting(
    settings: Settings,
) -> None:
    settings.MODEL_RAG_OUTPUT = "tests.recording.TrackedRecordingOutput"

    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        call_command("sync_model_rag")


def test_the_command_with_an_output_setting_without_a_backend_names_the_backend_key(
    settings: Settings,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"OPTIONS": {}}

    with pytest.raises(ImproperlyConfigured, match="BACKEND"):
        call_command("sync_model_rag")


def test_the_command_with_a_backend_that_cannot_be_imported_names_the_backend(
    settings: Settings,
) -> None:
    settings.MODEL_RAG_OUTPUT = {"BACKEND": "tests.recording.NoSuchOutput"}

    with pytest.raises(
        ImproperlyConfigured, match=r"tests\.recording\.NoSuchOutput"
    ) as excinfo:
        call_command("sync_model_rag")

    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.django_db
def test_the_command_with_a_backend_without_replace_names_it_before_running_a_model(
    settings: Settings, built_prune_only_outputs: list[PruneOnlyOutput]
) -> None:
    # Registered but without rows: running it would call prune alone, never
    # replace, so only a check made up front can notice the missing replace.
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": "tests.recording.PruneOnlyOutput"}

    with pytest.raises(ImproperlyConfigured) as excinfo:
        call_command("sync_model_rag")

    message = str(excinfo.value)
    assert "PruneOnlyOutput" in message
    assert "replace" in message
    # The check may come before or after the backend is built: either way,
    # nothing reaches it.
    assert [output.pruned for output in built_prune_only_outputs] in ([], [[]])


@pytest.mark.django_db
def test_the_command_runs_a_registered_model_into_the_configured_backend(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    category = Category.objects.create(name="Tools")
    hammer = Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": "tests.recording.TrackedRecordingOutput"}

    call_command("sync_model_rag")

    [output] = built_outputs
    assert [document.text for document in output.documents()] == ["Hammer"]
    assert output.pruned == [("testapp.product", {f"testapp.product:{hammer.pk}"})]
    assert output.calls == ["replace", "prune"]


@pytest.mark.django_db
def test_the_command_passes_the_output_options_to_the_backend_as_keyword_arguments(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": "tests.recording.TrackedRecordingOutput",
        "OPTIONS": {"collection": "catalog", "batch_size": 50},
    }

    call_command("sync_model_rag")

    [output] = built_outputs
    assert output.options == {"collection": "catalog", "batch_size": 50}
