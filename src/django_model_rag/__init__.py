"""Discover and normalize text content from any Django model."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

from django.db.models import Model

__all__ = ["NormalizedDocument", "SyncPipeline", "rag"]

_REPR_TEXT_LENGTH = 60
_FIELD_SEPARATOR = "\n\n"


def _short_repr(value: str) -> str:
    """Quote ``value``, cut to its first characters and marked with an ellipsis."""
    if len(value) > _REPR_TEXT_LENGTH:
        return repr(value[:_REPR_TEXT_LENGTH]) + "…"
    return repr(value)


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
        title = _short_repr(self.title)
        text = _short_repr(self.text)
        return f"<NormalizedDocument {self.source_key} title={title} text={text}>"


def _field_text(instance: Model, name: str) -> str:
    """Read the field ``name`` of ``instance`` as text."""
    return str(getattr(instance, name))


def _is_blank(text: str) -> bool:
    """Tell whether ``text`` holds nothing but whitespace."""
    return not text.strip()


def _document_text(instance: Model, fields: list[str]) -> str:
    """Join the non-blank values of the ``fields`` of ``instance``, in order."""
    return _FIELD_SEPARATOR.join(
        text for name in fields if not _is_blank(text := _field_text(instance, name))
    )


def _document_title(instance: Model, fields: list[str]) -> str:
    """Read the first of the ``fields`` of ``instance``, or nothing when it is blank."""
    title = _field_text(instance, fields[0])
    return "" if _is_blank(title) else title


class SyncPipeline:
    """Turn registered models into normalized documents."""

    def run(self) -> list[NormalizedDocument]:
        """Produce the documents of every registered model."""
        return [
            NormalizedDocument(
                text=text,
                source_app_label=model._meta.app_label,  # Django's public meta API
                source_model=model._meta.model_name or "",  # Django's public meta API
                source_pk=instance.pk,
                title=_document_title(instance, fields),
            )
            for model, fields in rag.declarations()
            for instance in model._default_manager.order_by("pk")
            if (text := _document_text(instance, fields))
        ]


class Registry:
    """Hold the models whose content feeds the pipeline."""

    def __init__(self) -> None:
        """Start with no registered model."""
        self._fields: dict[type[Model], list[str]] = {}

    def register(self, model: type[Model], *, fields: list[str]) -> None:
        """Register ``model`` with the fields to extract."""
        self._fields[model] = fields

    def declarations(self) -> list[tuple[type[Model], list[str]]]:
        """List each registered model with its declared fields."""
        return list(self._fields.items())

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``."""
        del self._fields[model]


rag = Registry()
