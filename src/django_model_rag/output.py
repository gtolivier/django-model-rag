"""The output protocol: where the pipeline hands over the documents it builds."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from django_model_rag.documents import NormalizedDocument


class DocumentOutput(Protocol):
    """Receives the documents of the pipeline, grouped by source key."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Replace the stored documents of each source key with its group."""

    def prune(self, model_label: str, kept_keys: set[str]) -> None:
        """Delete the documents of ``model_label`` whose source key is not kept."""
