"""The registry of models whose content feeds the pipeline."""

from dataclasses import dataclass

from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db.models import Model


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


def _require_content_fields(model: type[Model], fields: list[str]) -> None:
    """Fail unless ``fields`` is a non-empty list of ``model``'s content fields.

    Raises:
        ImproperlyConfigured: ``fields`` is a single string instead of a
            list, it is empty, or a field is not one of the model's or is a
            relation.
    """
    if isinstance(fields, str):
        message = f"{model.__name__}: fields must be a list of field names"
        raise ImproperlyConfigured(message)
    if not fields:
        message = f"{model.__name__} declares no field"
        raise ImproperlyConfigured(message)
    for name in fields:
        _require_content_field(model, name)


class AlreadyRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's AlreadyRegistered
    """A model is registered a second time."""


class NotRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's NotRegistered
    """A model that is not registered is unregistered."""


@dataclass(frozen=True)
class _Declaration:
    """What a model declares when it is registered."""

    fields: list[str]
    title_field: str | None


class Registry:
    """Hold the models whose content feeds the pipeline."""

    def __init__(self) -> None:
        """Start with no registered model."""
        self._declarations: dict[type[Model], _Declaration] = {}

    def register(
        self,
        model: type[Model],
        *,
        fields: list[str],
        title_field: str | None = None,
    ) -> None:
        """Register ``model`` with the fields to extract.

        ``title_field`` names the field whose value is the document title.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: ``fields`` is a single string instead of
                a list, no field is declared, or a field (``title_field``
                included) is not one of the model's or is a relation.
        """
        if model in self._declarations:
            message = f"{model.__name__} is already registered"
            raise AlreadyRegistered(message)
        _require_content_fields(model, fields)
        if title_field is not None:
            _require_content_field(model, title_field)
        self._declarations[model] = _Declaration(list(fields), title_field)

    def title_field(self, model: type[Model]) -> str | None:
        """Name the field declared as the title of ``model``, if any."""
        declaration = self._declarations.get(model)
        return None if declaration is None else declaration.title_field

    def declarations(self) -> list[tuple[type[Model], list[str]]]:
        """List each registered model with its declared fields."""
        return [
            (model, declaration.fields)
            for model, declaration in self._declarations.items()
        ]

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
