"""The output protocol: where the pipeline hands over the documents it builds."""

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Protocol

from django_model_rag.documents import NormalizedDocument


class DocumentOutput(Protocol):
    """Receives the documents of the pipeline, grouped by source key."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Replace the stored documents of each source key with its group."""

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Delete the documents of ``model_label`` whose source key is not kept."""


class ConsoleOutput:
    """Writes the documents to standard output."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Write each group: its source key, then its documents' titles and texts."""
        for key, documents in groups.items():
            if not documents:
                print(f"{key} removed")
                continue
            print(key)
            for document in documents:
                print(document.title)
                print(document.text)
