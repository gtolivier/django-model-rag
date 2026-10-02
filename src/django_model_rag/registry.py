"""The registry of models whose content feeds the pipeline."""

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


class AlreadyRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's AlreadyRegistered
    """A model is registered a second time."""


class NotRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's NotRegistered
    """A model that is not registered is unregistered."""


class Registry:
    """Hold the models whose content feeds the pipeline."""

    def __init__(self) -> None:
        """Start with no registered model."""
        self._fields: dict[type[Model], list[str]] = {}

    def register(self, model: type[Model], *, fields: list[str]) -> None:
        """Register ``model`` with the fields to extract.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: ``fields`` is a single string instead of
                a list, no field is declared, or a field is not one of the
                model's or is a relation.
        """
        if model in self._fields:
            message = f"{model.__name__} is already registered"
            raise AlreadyRegistered(message)
        if isinstance(fields, str):
            message = f"{model.__name__}: fields must be a list of field names"
            raise ImproperlyConfigured(message)
        if not fields:
            message = f"{model.__name__} declares no field"
            raise ImproperlyConfigured(message)
        for name in fields:
            _require_content_field(model, name)
        self._fields[model] = list(fields)

    def declarations(self) -> list[tuple[type[Model], list[str]]]:
        """List each registered model with its declared fields."""
        return list(self._fields.items())

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        if model not in self._fields:
            message = f"{model.__name__} is not registered"
            raise NotRegistered(message)
        del self._fields[model]


rag = Registry()
