"""An output that records what the pipeline hands it, and runs that read it back."""

from collections.abc import Mapping, Sequence

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

    def prune(self, model_label: str, kept_keys: set[str]) -> None:
        self.pruned.append((model_label, set(kept_keys)))
        self.calls.append("prune")

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
