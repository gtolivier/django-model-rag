import io
import re
from collections.abc import Iterator
from typing import TypeVar

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.core.management import CommandError, call_command
from django.db.models import Model
from pytest_django import Settings

from django_model_rag import BaseExtractor, NormalizedDocument, rag
from tests.recording import (
    PruneOnlyOutput,
    ReplaceOnlyOutput,
    TrackedRecordingOutput,
)
from tests.testapp.models import Category, Page, Product

M = TypeVar("M", bound=Model)
T = TypeVar("T")

# The dotted path of the backend whose built instances the tests read back.
TRACKED_BACKEND = "tests.recording.TrackedRecordingOutput"


def _only_built_during_the_test(built: list[T]) -> Iterator[list[T]]:
    """Yield a class's `built` list emptied, and empty it again after the test."""
    built.clear()
    yield built
    built.clear()


@pytest.fixture
def built_outputs() -> Iterator[list[TrackedRecordingOutput]]:
    """The TrackedRecordingOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(TrackedRecordingOutput.built)


@pytest.fixture
def built_prune_only_outputs() -> Iterator[list[PruneOnlyOutput]]:
    """The PruneOnlyOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(PruneOnlyOutput.built)


@pytest.fixture
def built_replace_only_outputs() -> Iterator[list[ReplaceOnlyOutput]]:
    """The ReplaceOnlyOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(ReplaceOnlyOutput.built)


def _create_a_hammer() -> Product:
    """Create a Hammer product, in a Tools category: one row to extract."""
    category = Category.objects.create(name="Tools")
    return Product.objects.create(
        name="Hammer", description="Drives nails.", price="9.90", category=category
    )


def test_the_command_without_an_output_setting_names_the_missing_setting() -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        call_command("sync_model_rag")


def test_the_command_with_an_output_setting_that_is_not_a_dict_names_the_setting(
    settings: Settings,
) -> None:
    settings.MODEL_RAG_OUTPUT = TRACKED_BACKEND

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
def test_the_command_with_a_backend_without_prune_names_it_before_running_a_model(
    settings: Settings, built_replace_only_outputs: list[ReplaceOnlyOutput]
) -> None:
    # Registered with a row: running it would call replace before prune, so
    # only a check made up front keeps the documents from reaching the output.
    _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": "tests.recording.ReplaceOnlyOutput"}

    with pytest.raises(ImproperlyConfigured) as excinfo:
        call_command("sync_model_rag")

    message = str(excinfo.value)
    assert "ReplaceOnlyOutput" in message
    assert "prune" in message
    # The check may come before or after the backend is built: either way,
    # nothing reaches it.
    assert [output.replaced for output in built_replace_only_outputs] in ([], [[]])


@pytest.mark.django_db
@pytest.mark.parametrize("bad_options", [None, ["collection", "catalog"]])
def test_the_command_with_options_that_are_not_a_dict_names_them_before_running_a_model(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    bad_options: object,
) -> None:
    # Registered with a row: running it would send the documents to the
    # output, so only a check made up front keeps them from reaching it.
    _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND, "OPTIONS": bad_options}

    with pytest.raises(ImproperlyConfigured, match="OPTIONS"):
        call_command("sync_model_rag")

    assert [output.calls for output in built_outputs] in ([], [[]])


@pytest.mark.django_db
@pytest.mark.parametrize("bad_backend", [TrackedRecordingOutput, None])
def test_the_command_with_a_backend_not_a_string_names_it_before_running_a_model(
    settings: Settings,
    built_outputs: list[TrackedRecordingOutput],
    bad_backend: object,
) -> None:
    # Registered with a row: running it would send the documents to the
    # output, so only a check made up front keeps them from reaching it.
    _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": bad_backend}

    with pytest.raises(ImproperlyConfigured, match="BACKEND"):
        call_command("sync_model_rag")

    assert [output.calls for output in built_outputs] in ([], [[]])


@pytest.mark.django_db
def test_the_command_runs_a_registered_model_into_the_configured_backend(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    hammer = _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    call_command("sync_model_rag")

    [output] = built_outputs
    assert [document.text for document in output.documents()] == ["Hammer"]
    assert output.pruned == [("testapp.product", {f"testapp.product:{hammer.pk}"})]
    assert output.calls == ["replace", "prune"]


@pytest.mark.django_db
def test_the_command_prunes_every_registered_model_in_registration_order(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    # Product registered before Category: the reverse of alphabetical and of
    # declaration order, so neither could pass for registration order.
    rag.register(Product, fields=["name"])
    rag.register(Category, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    call_command("sync_model_rag")

    [output] = built_outputs
    assert output.pruned == [("testapp.product", set()), ("testapp.category", set())]


@pytest.mark.django_db
def test_the_command_with_model_labels_prunes_only_those_models_in_the_given_order(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    # Named in the reverse of registration order, with a third registered
    # model left out: running every model, or in registration order, fails.
    rag.register(Category, fields=["name"])
    rag.register(Product, fields=["name"])
    rag.register(Page, fields=["title"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    call_command("sync_model_rag", "testapp.product", "testapp.category")

    [output] = built_outputs
    assert output.pruned == [("testapp.product", set()), ("testapp.category", set())]


@pytest.mark.django_db
@pytest.mark.parametrize("bad_label", ["testapp.nosuchmodel", "product"])
def test_the_command_with_a_label_naming_no_model_names_it_before_running_a_model(
    settings: Settings, built_outputs: list[TrackedRecordingOutput], bad_label: str
) -> None:
    # The bad label comes after a valid one with a row: resolving the labels
    # one by one while running would send that model's documents first.
    _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    with pytest.raises(CommandError, match=re.escape(bad_label)):
        call_command("sync_model_rag", "testapp.product", bad_label)

    # The check may come before or after the backend is built: either way,
    # nothing reaches it.
    assert [output.calls for output in built_outputs] in ([], [[]])


@pytest.mark.django_db
def test_the_command_with_a_label_naming_an_unregistered_model_names_it_up_front(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    # Category exists but is not registered, and comes after a registered
    # model with a row: running the models before checking them would send
    # Product's documents first.
    _create_a_hammer()
    rag.register(Product, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    with pytest.raises(CommandError, match=re.escape("testapp.category")):
        call_command("sync_model_rag", "testapp.product", "testapp.category")

    # The check may come before or after the backend is built: either way,
    # nothing reaches it.
    assert [output.calls for output in built_outputs] in ([], [[]])


@pytest.mark.django_db
@pytest.mark.usefixtures("built_outputs")
def test_the_command_writes_one_synced_line_per_model_in_run_order(
    settings: Settings,
) -> None:
    # Product registered before Category: a constant line, or one in
    # alphabetical order, cannot pass.
    rag.register(Product, fields=["name"])
    rag.register(Category, fields=["name"])
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}
    stdout = io.StringIO()

    call_command("sync_model_rag", stdout=stdout)

    assert stdout.getvalue().splitlines() == [
        "testapp.product: synced",
        "testapp.category: synced",
    ]


def _register_a_failing_extractor(model: type[M]) -> None:
    """Register for `model` an extractor whose extraction always raises.

    It raises ``RuntimeError("cannot extract <instance>")``: the run of
    `model` fails only if it has a row to extract.
    """

    @rag.register_extractor(model)
    class FailingExtractor(BaseExtractor[M]):
        def extract(self, instance: M) -> NormalizedDocument:
            message = f"cannot extract {instance}"
            raise RuntimeError(message)


def _register_a_failing_category_then_product() -> None:
    """Register Category, whose run raises, then Product, whose run succeeds.

    Category's extraction raises ``RuntimeError("cannot extract Tools")``: a
    Category row named Tools makes that extraction actually run.
    """
    Category.objects.create(name="Tools")
    _register_a_failing_extractor(Category)
    rag.register(Product, fields=["name"])


@pytest.mark.django_db
def test_the_command_goes_on_after_a_failed_model_then_fails_naming_it(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    # Category fails and is registered first: stopping at the failure would
    # leave Product unrun.
    _register_a_failing_category_then_product()
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    with pytest.raises(CommandError, match=re.escape("testapp.category")):
        call_command("sync_model_rag")

    [output] = built_outputs
    assert output.pruned == [("testapp.product", set())]


@pytest.mark.django_db
@pytest.mark.usefixtures("built_outputs")
def test_the_command_fails_naming_every_failed_model_in_run_order(
    settings: Settings,
) -> None:
    # Category and Page fail, around Product, which succeeds: naming only the
    # first or the last failure, or every model run, cannot pass.
    _register_a_failing_category_then_product()
    Page.objects.create(title="Home", slug="home")
    _register_a_failing_extractor(Page)
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}

    with pytest.raises(CommandError) as excinfo:
        call_command("sync_model_rag", stdout=io.StringIO(), stderr=io.StringIO())

    message = str(excinfo.value)
    assert "testapp.category" in message
    assert "testapp.page" in message
    assert message.index("testapp.category") < message.index("testapp.page")
    assert "testapp.product" not in message


@pytest.mark.django_db
@pytest.mark.usefixtures("built_outputs")
def test_the_command_writes_a_failed_model_and_its_error_on_stderr_then_goes_on(
    settings: Settings,
) -> None:
    # Category fails and is registered first: Product, synced after it, shows
    # the command went on past the failure instead of reporting it only at
    # the end.
    _register_a_failing_category_then_product()
    settings.MODEL_RAG_OUTPUT = {"BACKEND": TRACKED_BACKEND}
    stdout = io.StringIO()
    stderr = io.StringIO()

    with pytest.raises(CommandError):
        call_command("sync_model_rag", stdout=stdout, stderr=stderr)

    assert stderr.getvalue().splitlines() == [
        "testapp.category: RuntimeError: cannot extract Tools"
    ]
    assert stdout.getvalue().splitlines() == ["testapp.product: synced"]


@pytest.mark.django_db
def test_the_command_passes_the_output_options_to_the_backend_as_keyword_arguments(
    settings: Settings, built_outputs: list[TrackedRecordingOutput]
) -> None:
    settings.MODEL_RAG_OUTPUT = {
        "BACKEND": TRACKED_BACKEND,
        "OPTIONS": {"collection": "catalog", "batch_size": 50},
    }

    call_command("sync_model_rag")

    [output] = built_outputs
    assert output.options == {"collection": "catalog", "batch_size": 50}
