"""Discover and normalize text content from any Django model."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NormalizedDocument"]

REPR_TEXT_LENGTH = 60


@dataclass(kw_only=True, repr=False)
class NormalizedDocument:
    """A piece of text together with the model instance it comes from."""

    text: str
    source_app_label: str
    source_model: str
    source_pk: object
    title: str = ""
    url: str = ""
    language: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_key(self) -> str:
        """Identify the source instance as ``app_label.model:pk``."""
        return f"{self.source_app_label}.{self.source_model}:{self.source_pk}"

    def __repr__(self) -> str:
        text = self.text
        if len(text) > REPR_TEXT_LENGTH:
            text = text[:REPR_TEXT_LENGTH] + "…"
        return (
            f"<NormalizedDocument {self.source_key} title={self.title!r} text={text!r}>"
        )
