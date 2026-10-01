"""Discover and normalize text content from any Django model."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["NormalizedDocument"]


@dataclass(kw_only=True)
class NormalizedDocument:
    """A piece of text together with the model instance it comes from."""

    text: str
    source_app_label: str
    source_model: str
    source_pk: int
    title: str = ""
    url: str = ""
    language: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
