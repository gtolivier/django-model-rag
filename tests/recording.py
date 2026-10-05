"""An output that records what the pipeline hands it, and runs that read it back."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import ClassVar

from django.db.models import Model

from django_model_rag import NormalizedDocument, SyncPipeline


class RecordingOutput:
    """An output that records what the pipeline hands it."""

    def __init__(self) -> None:
        self.replaced: list[Mapping[str, Sequence[NormalizedDocument]]] = []
        self.pruned: list[tuple[str, set[str]]] = []
        # The name of each method called, in call order: replaced and pruned
        # alone cannot tell whether a prune came before or after a replace.
        self.calls: list[str] = []

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        # Copied, so that a pipeline reusing its mapping between calls cannot
        # rewrite what was already recorded.
        self.replaced.append({key: list(group) for key, group in groups.items()})
        self.calls.append("replace")

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        self.pruned.append((model_label, set(kept_keys)))
        self.calls.append("prune")

    def received_groups(self) -> dict[str, list[NormalizedDocument]]:
        """Every group received, merged into one mapping across replace calls.

        run() may split a model's groups across several calls.
        """
        return {
            source_key: list(group)
            for groups in self.replaced
            for source_key, group in groups.items()
        }

    def documents(self) -> list[NormalizedDocument]:
        """Every document received, flattened in the order received.

        Groups in the order they came, each group's documents in its order.
        """
        return [
            document
            for groups in self.replaced
            for group in groups.values()
            for document in group
        ]


class TrackedRecordingOutput(RecordingOutput):
    """A recording output that keeps every instance built of it.

    For code that builds its output itself from a dotted path, such as the
    management command: the test reads back the instance it built, and the
    keyword arguments it was built with.
    """

    built: ClassVar[list["TrackedRecordingOutput"]] = []

    def __init__(self, **options: object) -> None:
        super().__init__()
        self.options = options
        TrackedRecordingOutput.built.append(self)


# The dotted path of the backend whose built instances the tests read back.
TRACKED_BACKEND = "tests.recording.TrackedRecordingOutput"


class FailingReplaceError(Exception):
    """Raised by FailingOnKeyOutput on the group of its failing source key."""


class FailingOnKeyOutput(TrackedRecordingOutput):
    """A tracked recording output whose replace fails on one source key.

    A replace call holding the group of ``failing_source_key`` raises before
    recording anything; every other call is recorded as usual.
    """

    def __init__(self, failing_source_key: str, **options: object) -> None:
        super().__init__(**options)
        self.failing_source_key = failing_source_key

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        if self.failing_source_key in groups:
            raise FailingReplaceError(self.failing_source_key)
        super().replace(groups)


# The dotted path of the tracked backend that fails on one source key.
FAILING_ON_KEY_BACKEND = "tests.recording.FailingOnKeyOutput"


class PruneOnlyOutput:
    """A broken output backend: a callable prune, but no replace at all.

    Keeps every instance built of it and every prune call it receives, so a
    test can tell that nothing was sent to it.
    """

    built: ClassVar[list["PruneOnlyOutput"]] = []

    def __init__(self) -> None:
        self.pruned: list[tuple[str, set[str]]] = []
        PruneOnlyOutput.built.append(self)

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        self.pruned.append((model_label, set(kept_keys)))


class ReplaceOnlyOutput:
    """A broken output backend: a callable replace, but no prune at all.

    Keeps every instance built of it and every replace call it receives, so a
    test can tell that nothing was sent to it.
    """

    built: ClassVar[list["ReplaceOnlyOutput"]] = []

    def __init__(self) -> None:
        self.replaced: list[Mapping[str, Sequence[NormalizedDocument]]] = []
        ReplaceOnlyOutput.built.append(self)

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        self.replaced.append({key: list(group) for key, group in groups.items()})


# Not a class but an instance, with a callable replace and prune: a dotted
# path to it names a misconfigured output backend. A test reads its calls
# back to tell that nothing was sent to it.
RECORDING_OUTPUT_INSTANCE = RecordingOutput()


def run_documents(
    models: Sequence[type[Model]] | None = None,
) -> list[NormalizedDocument]:
    """Run ``models`` into a recording output and return the documents it received."""
    output = RecordingOutput()
    SyncPipeline(output).run(models)
    return output.documents()


def run_instance_documents(instance: Model) -> list[NormalizedDocument]:
    """Run ``instance`` into a recording output and return the documents it received."""
    output = RecordingOutput()
    SyncPipeline(output).run_instance(instance)
    return output.documents()
