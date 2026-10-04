"""The normalized document: a piece of text and the model instance it comes from."""

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

_REPR_TEXT_LENGTH = 60


def _short_repr(value: str) -> str:
    """Quote ``value``, cut to its first characters and marked with an ellipsis."""
    if len(value) > _REPR_TEXT_LENGTH:
        return repr(value[:_REPR_TEXT_LENGTH]) + "…"
    return repr(value)


def _frozen_permissions(permissions: Iterable[str]) -> frozenset[str]:
    """Return ``permissions`` as a frozenset; raise TypeError unless all are strings."""
    if isinstance(permissions, str):
        msg = "permissions must be a collection of strings, not a bare string"
        raise TypeError(msg)
    granted = frozenset(permissions)
    if not all(isinstance(item, str) for item in granted):
        msg = "permissions must be a collection of strings"
        raise TypeError(msg)
    return granted


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
    permissions: AbstractSet[str] = frozenset()

    # Compare by value, but keep documents out of sets and dict keys: a frozen
    # dataclass would otherwise generate a hash. ClassVar[None] lets mypy see
    # the class itself as unhashable, not only its instances.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]  # typeshed types object.__hash__ as a method

    def __post_init__(self) -> None:
        """Validate source_pk and permissions; store frozen metadata and permissions."""
        if self.source_pk is None:
            msg = "source_pk cannot be None"
            raise ValueError(msg)
        permissions = _frozen_permissions(self.permissions)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "permissions", permissions)

    @property
    def source_key(self) -> str:
        """Identify the source instance as ``app_label.model:pk``."""
        return f"{self.source_app_label}.{self.source_model}:{self.source_pk}"

    def __repr__(self) -> str:
        """Show the source key, the title and the text, truncated to stay short."""
        title = _short_repr(self.title)
        text = _short_repr(self.text)
        return f"<NormalizedDocument {self.source_key} title={title} text={text}>"
