"""The registry of models whose content feeds the pipeline."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db.models import Model

from django_model_rag.extractors import BaseExtractor

ExtractorClass = TypeVar("ExtractorClass", bound=BaseExtractor[Any])

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


def _require_extractor_class(extractor_class: type) -> None:
    """Fail unless ``extractor_class`` is a concrete ``BaseExtractor``.

    Raises:
        ImproperlyConfigured: ``extractor_class`` does not derive from
            ``BaseExtractor``, or does not implement ``extract``.
    """
    if not issubclass(extractor_class, BaseExtractor):
        message = f"{extractor_class.__name__} must derive from BaseExtractor"
        raise ImproperlyConfigured(message)
    if inspect.isabstract(extractor_class):
        message = f"{extractor_class.__name__} does not implement extract"
        raise ImproperlyConfigured(message)


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
        # One dict for both kinds keeps the registration order across them
        self._registrations: dict[type[Model], Declaration | BaseExtractor[Any]] = {}

    def registered_models(self) -> list[type[Model]]:
        """List the registered models, in registration order."""
        return list(self._registrations)

    def _is_registered(self, model: type[Model]) -> bool:
        """Tell whether ``model`` is registered with fields or an extractor."""
        return model in self._registrations

    def _require_unregistered(self, model: type[Model]) -> None:
        """Fail if ``model`` is already registered with fields or an extractor.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
        """
        if self._is_registered(model):
            message = f"{model.__name__} is already registered"
            raise AlreadyRegistered(message)

    def require_registered(self, model: type[Model]) -> None:
        """Fail unless ``model`` is registered with fields or an extractor.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        if not self._is_registered(model):
            message = f"{model.__name__} is not registered"
            raise NotRegistered(message)

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
        self._require_unregistered(model)
        _require_content_fields(model, fields)
        if title_field is not None:
            _require_content_field(model, title_field)
        self._registrations[model] = Declaration(tuple(fields), title_field)

    def register_extractor(
        self, model: type[Model]
    ) -> Callable[[type[ExtractorClass]], type[ExtractorClass]]:
        """Register the decorated extractor class as the one of ``model``.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: the decorated class does not derive from
                ``BaseExtractor``, or does not implement ``extract``.
        """

        def decorator(extractor_class: type[ExtractorClass]) -> type[ExtractorClass]:
            _require_extractor_class(extractor_class)
            self._require_unregistered(model)
            self._registrations[model] = extractor_class()
            return extractor_class

        return decorator

    def extractors(self) -> list[tuple[type[Model], BaseExtractor[Any]]]:
        """List each model with the extractor registered for it."""
        return [
            (model, registration)
            for model, registration in self._registrations.items()
            if isinstance(registration, BaseExtractor)
        ]

    def declarations(self) -> list[tuple[type[Model], Declaration]]:
        """List each registered model with what it declared."""
        return [
            (model, registration)
            for model, registration in self._registrations.items()
            if isinstance(registration, Declaration)
        ]

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        self.require_registered(model)
        del self._registrations[model]


rag = Registry()
