"""Discover and normalize text content from any Django model."""

from dataclasses import dataclass

__all__ = ["NormalizedDocument"]


@dataclass
class NormalizedDocument:
    """A piece of text together with the model instance it comes from."""

    text: str
    source_app_label: str
    source_model: str
    source_pk: int
