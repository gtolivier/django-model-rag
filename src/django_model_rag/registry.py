"""The registry of models whose content feeds the pipeline."""

from dataclasses import dataclass
from typing import TypeAlias

from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db.models import Model

FieldNames: TypeAlias = list[str] | tuple[str, ...]
"""The field names a model declares: a list or a tuple, never a bare string."""


def _require_content_field(model: type[Model], name: str) -> None:
    """Fail unless ``model`` has a non-relation field called ``name``.

    Raises:
        ImproperlyConfigured: ``model`` has no such field, or it is a
            relation.
    """
    try:
        field = model._meta.get_field(name)
    except FieldDoesNotExist as error:
        message = f"{model.__name__} has no field {name!r}"
        raise ImproperlyConfigured(message) from error
    if field.is_relation:
        message = f"{model.__name__}.{name} is a relation, not a content field"
        raise ImproperlyConfigured(message)


def _require_content_fields(model: type[Model], fields: FieldNames) -> None:
    """Fail unless ``fields`` is a non-empty list or tuple of content fields.

    Raises:
        ImproperlyConfigured: ``fields`` is not a list or a tuple (a bare
            string or a set, say), it is empty, a field is declared twice,
            or a field is not one of ``model``'s or is a relation.
    """
    if not isinstance(fields, list | tuple):
        message = f"{model.__name__}: fields must be a list or a tuple of field names"
        raise ImproperlyConfigured(message)
    if not fields:
        message = f"{model.__name__} declares no field"
        raise ImproperlyConfigured(message)
    seen: set[str] = set()
    for name in fields:
        if name in seen:
            message = f"{model.__name__}: field {name!r} is declared twice"
            raise ImproperlyConfigured(message)
        seen.add(name)
        _require_content_field(model, name)


class AlreadyRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's AlreadyRegistered
    """A model is registered a second time."""


class NotRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's NotRegistered
    """A model that is not registered is unregistered."""


@dataclass(frozen=True)
class Declaration:
    """What a model declares when it is registered."""

    fields: tuple[str, ...]
    title_field: str | None


class Registry:
    """Hold the models whose content feeds the pipeline."""

    def __init__(self) -> None:
        """Start with no registered model."""
        self._declarations: dict[type[Model], Declaration] = {}

    def register(
        self,
        model: type[Model],
        *,
        fields: FieldNames,
        title_field: str | None = None,
    ) -> None:
        """Register ``model`` with the fields to extract.

        ``title_field`` names the field whose value is the document title.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: ``fields`` is not a list or a tuple, no
                field is declared, a field is declared twice, or a field
                (``title_field`` included) is not one of the model's or is a
                relation.
        """
        if model in self._declarations:
            message = f"{model.__name__} is already registered"
            raise AlreadyRegistered(message)
        _require_content_fields(model, fields)
        if title_field is not None:
            _require_content_field(model, title_field)
        self._declarations[model] = Declaration(tuple(fields), title_field)

    def declarations(self) -> list[tuple[type[Model], Declaration]]:
        """List each registered model with what it declared."""
        return list(self._declarations.items())

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        if model not in self._declarations:
            message = f"{model.__name__} is not registered"
            raise NotRegistered(message)
        del self._declarations[model]


rag = Registry()
