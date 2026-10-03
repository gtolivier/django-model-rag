"""The registry of models whose content feeds the pipeline."""

import inspect
from collections.abc import Callable
from typing import Any, TypeAlias

from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db.models import Model

from django_model_rag.extractors import (
    BaseExtractor,
    DeclaredFieldsExtractor,
    M,
    text_fields,
)

FieldNames: TypeAlias = list[str] | tuple[str, ...]
"""The field names a model declares: a list or a tuple, never a bare string."""


def _guessed_fields(model: type[Model], exclude: FieldNames) -> list[str]:
    """List ``model``'s text fields, except those named in ``exclude``.

    Raises:
        ImproperlyConfigured: ``model`` has no text field, or ``exclude``
            names all of them.
    """
    names = text_fields(model)
    if not names:
        message = f"{model.__name__} has no text field to guess"
        raise ImproperlyConfigured(message)
    guessed = [name for name in names if name not in exclude]
    if not guessed:
        message = f"{model.__name__} has no text field left to extract"
        raise ImproperlyConfigured(message)
    return guessed


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


def _require_relation_accessors(model: type[Model], names: FieldNames) -> None:
    """Fail unless each of ``names`` is a relation accessor of ``model``.

    Raises:
        ImproperlyConfigured: a name to follow is not a relation accessor.
    """
    accessors = {
        field.get_accessor_name() if field.auto_created else field.name  # type: ignore[union-attr] # reverse relations have it; get_fields() is typed too loosely
        for field in model._meta.get_fields()
        if field.is_relation
    }
    for name in names:
        if name not in accessors:
            message = f"{model.__name__}: cannot follow {name!r}, not a relation"
            raise ImproperlyConfigured(message)


def _require_field_names(model: type[Model], names: object, argument: str) -> None:
    """Fail unless ``names``, given as ``argument``, is a list or a tuple.

    Raises:
        ImproperlyConfigured: ``names`` is not a list or a tuple (a bare
            string or a set, say).
    """
    if not isinstance(names, list | tuple):
        message = (
            f"{model.__name__}: {argument} must be a list or a tuple of field names"
        )
        raise ImproperlyConfigured(message)


def _require_content_fields(model: type[Model], fields: FieldNames) -> None:
    """Fail unless ``fields`` is a non-empty list or tuple of content fields.

    Raises:
        ImproperlyConfigured: ``fields`` is not a list or a tuple (a bare
            string or a set, say), it is empty, a field is declared twice,
            or a field is not one of ``model``'s or is a relation.
    """
    _require_field_names(model, fields, "fields")
    if not fields:
        message = f"{model.__name__} declares no field"
        raise ImproperlyConfigured(message)
    _require_distinct_content_fields(model, fields, "declared")


def _require_distinct_content_fields(
    model: type[Model], names: FieldNames, verb: str
) -> None:
    """Fail unless ``names`` names each of ``model``'s content fields once.

    ``verb`` says what the names are for, in the error: declared, excluded.

    Raises:
        ImproperlyConfigured: a name is not one of ``model``'s fields or is a
            relation, or it is given twice.
    """
    seen: set[str] = set()
    for name in names:
        if name in seen:
            message = f"{model.__name__}: field {name!r} is {verb} twice"
            raise ImproperlyConfigured(message)
        seen.add(name)
        _require_content_field(model, name)


def _require_fields_or_exclude(
    model: type[Model], fields: FieldNames | None, exclude: FieldNames
) -> None:
    """Fail if both ``fields`` and ``exclude`` are given.

    Raises:
        ImproperlyConfigured: ``exclude`` is combined with declared fields.
    """
    if fields is not None and exclude:
        message = f"{model.__name__}: exclude cannot be combined with declared fields"
        raise ImproperlyConfigured(message)


def _require_extractor_class(extractor_class: Callable[..., object]) -> None:
    """Fail unless ``extractor_class`` is a concrete ``BaseExtractor``.

    Raises:
        ImproperlyConfigured: ``extractor_class`` is not a class deriving
            from ``BaseExtractor`` (a plain function, say), or does not
            implement ``extract``.
    """
    if not inspect.isclass(extractor_class) or not issubclass(
        extractor_class, BaseExtractor
    ):
        message = f"{extractor_class.__name__} must derive from BaseExtractor"
        raise ImproperlyConfigured(message)
    if inspect.isabstract(extractor_class):
        message = f"{extractor_class.__name__} does not implement extract"
        raise ImproperlyConfigured(message)


class AlreadyRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's AlreadyRegistered
    """A model is registered a second time."""


class NotRegistered(Exception):  # noqa: N818 - public name mirrors Django admin's NotRegistered
    """A model that is not registered is unregistered."""


class Registry:
    """Hold the models whose content feeds the pipeline."""

    def __init__(self) -> None:
        """Start with no registered model."""
        # Declared fields are an extractor too: one dict keeps the
        # registration order across both kinds. It holds factories, not
        # extractors, so that no state an extractor keeps leaks between runs.
        self._registrations: dict[type[Model], Callable[[], BaseExtractor[Any]]] = {}

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

    def _require_registered(self, model: type[Model]) -> None:
        """Fail unless ``model`` is registered with fields or an extractor.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        if not self._is_registered(model):
            message = f"{model.__name__} is not registered"
            raise NotRegistered(message)

    def new_extractor(self, model: type[Model]) -> BaseExtractor[Any]:
        """Build a fresh instance of the extractor ``model`` is registered with.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        self._require_registered(model)
        return self._registrations[model]()

    def register(
        self,
        model: type[Model],
        *,
        fields: FieldNames | None = None,
        title_field: str | None = None,
        exclude: FieldNames = (),
        follow: FieldNames = (),
    ) -> None:
        """Register ``model`` with the fields to extract.

        Without ``fields``, the model's text fields are extracted, except
        those named in ``exclude``.
        ``title_field`` names the field whose value is the document title.
        The text of the relations named in ``follow`` comes after the fields.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: ``fields`` or ``exclude`` is not a list or
                a tuple, no field is declared, a field is declared or excluded
                twice, or a field (``title_field`` and ``exclude`` included) is
                not one of the model's or is a relation, ``exclude`` is
                combined with ``fields``, or, without ``fields``, the model
                has no text field or ``exclude`` names all of them.
        """
        self._require_unregistered(model)
        _require_field_names(model, exclude, "exclude")
        _require_fields_or_exclude(model, fields, exclude)
        if fields is None:
            _require_distinct_content_fields(model, exclude, "excluded")
            fields = _guessed_fields(model, exclude)
        else:
            _require_content_fields(model, fields)
        if title_field is not None:
            _require_content_field(model, title_field)
        if follow:
            _require_relation_accessors(model, follow)
        declared = tuple(fields)
        followed = tuple(follow)
        self._registrations[model] = lambda: DeclaredFieldsExtractor(
            declared, title_field, followed
        )

    def register_extractor(
        self, model: type[M]
    ) -> Callable[[type[BaseExtractor[M]]], type[BaseExtractor[Any]]]:
        """Register the decorated extractor class as the one of ``model``.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: the decorated class does not derive from
                ``BaseExtractor``, or does not implement ``extract``.
        """

        def decorator(
            extractor_class: type[BaseExtractor[M]],
        ) -> type[BaseExtractor[M]]:
            _require_extractor_class(extractor_class)
            self._require_unregistered(model)
            self._registrations[model] = extractor_class
            return extractor_class

        return decorator

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        self._require_registered(model)
        del self._registrations[model]


rag = Registry()
