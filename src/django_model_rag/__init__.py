"""Discover and normalize text content from any Django model."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

__all__ = ["NormalizedDocument"]

_REPR_TEXT_LENGTH = 60


def _shorten(value: str) -> str:
    """Cut ``value`` to its first characters, marking the cut with an ellipsis."""
    if len(value) > _REPR_TEXT_LENGTH:
        return value[:_REPR_TEXT_LENGTH] + "…"
    return value


@dataclass(frozen=True, kw_only=True, repr=False)
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

    # Compare by value, but keep documents out of sets and dict keys: a frozen
    # dataclass would otherwise generate a hash. ClassVar[None] lets mypy see
    # the class itself as unhashable, not only its instances.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]  # typeshed types object.__hash__ as a method

    def __post_init__(self) -> None:
        """Require a source primary key; copy the metadata so callers can't alter it."""
        if self.source_pk is None:
            msg = "source_pk cannot be None"
            raise ValueError(msg)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def source_key(self) -> str:
        """Identify the source instance as ``app_label.model:pk``."""
        return f"{self.source_app_label}.{self.source_model}:{self.source_pk}"

    def __repr__(self) -> str:
        """Show the source key, the title and the text, truncated to stay short."""
        title = _shorten(self.title)
        text = _shorten(self.text)
        return f"<NormalizedDocument {self.source_key} title={title!r} text={text!r}>"
