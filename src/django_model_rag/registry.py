"""The registry of models whose content feeds the pipeline."""

import inspect
from collections.abc import Callable
from typing import Any, TypeAlias

from django.apps import apps
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.db.models import Field, ForeignKey, ForeignObjectRel, Model
from django.db.models.constants import LOOKUP_SEP
from django.db.models.signals import post_delete, pre_delete

from django_model_rag.apps import DISCOVERED_MODULE
from django_model_rag.extractors import (
    BaseExtractor,
    DeclaredFieldsExtractor,
    M,
    PathLink,
    guessed_language_field,
    path_links,
    relations_by_accessor,
    text_fields,
)

FieldNames: TypeAlias = list[str] | tuple[str, ...]
"""The field names a model declares: a list or a tuple, never a bare string."""

PermissionNames: TypeAlias = list[str] | tuple[str, ...]
"""The ``app_label.codename`` permissions a model's documents require."""


def _language_field(
    model: type[Model], language_field: str | None, language: str | None
) -> str | None:
    """Name the field ``model``'s documents read their language from, if any.

    It is the declared ``language_field``; without it, an own field guessed
    by name, unless the language is the constant ``language``.
    """
    if language_field is not None or language is not None:
        return language_field
    return guessed_language_field(model)


def _guessed_fields(
    model: type[Model],
    exclude: FieldNames,
    follow: FieldNames,
    metadata_fields: tuple[str | None, ...],
) -> list[str]:
    """List ``model``'s text fields, except those named in ``exclude``.

    The ``metadata_fields`` are left out too.

    A model that follows relations may be left with no text field of its
    own: their text is enough to make a document.

    Raises:
        ImproperlyConfigured: ``model`` follows no relation and has no text
            field, or none left once ``exclude`` is applied.
    """
    names = [name for name in text_fields(model) if name not in metadata_fields]
    if not names and not follow:
        message = f"{model.__name__} has no text field to guess"
        raise ImproperlyConfigured(message)
    guessed = [name for name in names if name not in exclude]
    if not guessed and not follow:
        message = f"{model.__name__} has no text field left to extract"
        raise ImproperlyConfigured(message)
    return guessed


def _model_through(link: PathLink, path: str) -> type[Model]:
    """Return the model ``link``, a link of ``path``, leads to.

    Raises:
        ImproperlyConfigured: ``link`` is not a relation, is a generic
            foreign key, or holds several objects.
    """
    relation = link.relation
    related_model = relation.related_model
    origin = f"{link.model.__name__}.{link.name}"
    if related_model is None:
        # A generic foreign key is a relation, only one without a single model.
        kind = "a generic foreign key" if relation.is_relation else "not a relation"
        message = f"{origin} is {kind}, so {path!r} cannot go through it"
        raise ImproperlyConfigured(message)
    if relation.many_to_many or relation.one_to_many:
        message = f"{origin} holds several objects, so {path!r} has no single value"
        raise ImproperlyConfigured(message)
    return related_model


def _require_content_field(model: type[Model], path: str) -> None:
    """Fail unless ``model`` has a non-relation field at ``path``.

    ``path`` is a field name, or a lookup path such as ``category__name``:
    the field is then looked up on the related model it leads to.

    Raises:
        ImproperlyConfigured: ``model`` has no such field, or it is a
            relation; or a link of ``path`` is not a relation; or ``path``
            goes through a relation while models are loading.
    """
    *steps, field_name = path.split(LOOKUP_SEP)
    if steps:
        _require_models_ready(model, f"resolve the lookup path {path!r}")
    target = model
    try:
        for link in path_links(model, path):
            target = _model_through(link, path)
        field = target._meta.get_field(field_name)
    except FieldDoesNotExist as error:
        message = f"{model.__name__} has no field {path!r}"
        raise ImproperlyConfigured(message) from error
    if field.is_relation:
        message = (
            f"{path!r} ends on {target.__name__}.{field_name}, "
            "a relation, not a content field"
        )
        raise ImproperlyConfigured(message)


def _require_models_ready(model: type[Model], action: str) -> None:
    """Fail unless every model is loaded, as ``action`` on ``model`` needs.

    Raises:
        ImproperlyConfigured: models are still loading (``model`` is
            registered from a models module, say).
    """
    if not apps.models_ready:
        message = (
            f"{model.__name__}: cannot {action} while models are loading; "
            f"register from the app's {DISCOVERED_MODULE}.py module, which "
            "django_model_rag imports in its AppConfig.ready()"
        )
        raise ImproperlyConfigured(message)


def _require_followable_relations(model: type[Model], names: FieldNames) -> None:
    """Fail unless ``names`` names, once each, relations of ``model`` with text.

    Raises:
        ImproperlyConfigured: a name to follow is not a relation accessor, it
            is given twice, or its related model is unknown (a generic foreign
            key) or has no text field.
    """
    accessors = relations_by_accessor(model)
    seen: set[str] = set()
    for name in names:
        if name not in accessors:
            message = f"{model.__name__}: cannot follow {name!r}, not a relation"
            raise ImproperlyConfigured(message)
        if name in seen:
            message = f"{model.__name__}: relation {name!r} is followed twice"
            raise ImproperlyConfigured(message)
        seen.add(name)
        _require_related_text(model, name, accessors[name].related_model)


def _require_related_text(
    model: type[Model], name: str, related: type[Model] | None
) -> None:
    """Fail unless ``related``, followed from ``model`` as ``name``, has text.

    Raises:
        ImproperlyConfigured: ``related`` is unknown (``None``, for a generic
            foreign key) or has no text field.
    """
    if related is None:
        message = (
            f"{model.__name__}: cannot follow {name!r}, "
            "a generic foreign key has no single related model"
        )
        raise ImproperlyConfigured(message)
    if not text_fields(related):
        message = (
            f"{model.__name__}: cannot follow {name!r}, "
            f"{related.__name__} has no text field"
        )
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


def _require_own_fields(model: type[Model], names: FieldNames) -> None:
    """Fail unless each of ``names`` is a field of ``model`` itself.

    Raises:
        ImproperlyConfigured: a name is a lookup path through a relation.
    """
    for name in names:
        if LOOKUP_SEP in name:
            message = f"{model.__name__}: {name!r} is not an own field of the model"
            raise ImproperlyConfigured(message)


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


def _require_single_language_source(
    language: str | None, language_field: str | None
) -> None:
    """Fail if both a constant ``language`` and a ``language_field`` are given.

    Raises:
        ImproperlyConfigured: ``language`` is combined with ``language_field``.
    """
    if language is not None and language_field is not None:
        message = "language and language_field cannot be combined"
        raise ImproperlyConfigured(message)


def _require_text_language(language: object) -> None:
    """Fail if a constant ``language`` is given but is not a non-blank string.

    Raises:
        ImproperlyConfigured: ``language`` is blank or not a string.
    """
    if language is not None and (not isinstance(language, str) or not language.strip()):
        message = "language must be a non-blank string"
        raise ImproperlyConfigured(message)


def _require_permission_names(permissions: object) -> None:
    """Fail unless ``permissions`` is a list or a tuple of permission names.

    Raises:
        ImproperlyConfigured: ``permissions`` is not a list or a tuple (a
            bare string or a set, say), or one of its items is not a
            permission name.
    """
    if not isinstance(permissions, list | tuple):
        message = "permissions must be a list or a tuple of strings"
        raise ImproperlyConfigured(message)
    for permission in permissions:
        _require_permission_name(permission)


def _require_permission_name(permission: object) -> None:
    """Fail unless ``permission`` is a string of the form ``app_label.codename``.

    Raises:
        ImproperlyConfigured: ``permission`` is not a string, or its dot,
            app label or codename is missing.
    """
    if not isinstance(permission, str):
        message = f"permissions must be strings, not {permission!r}"
        raise ImproperlyConfigured(message)
    app_label, dot, codename = permission.partition(".")
    if not (app_label and dot and codename):
        message = f"permission {permission!r} is not of the form 'app_label.codename'"
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


def _delete_uid(model: type[Model]) -> str:
    """Name the post_delete connection of ``model``."""
    return f"django_model_rag.sync_delete.{model._meta.label}"


def concrete_model_of(model: type[Model]) -> type[Model]:
    """Return the model whose table holds the rows of ``model``."""
    # A proxy sends signals under its own sender: it is its concrete model. The
    # fallback only satisfies the stubs: Django sets it on every model class.
    return model._meta.concrete_model or model


def _model_and_proxies(model: type[Model]) -> list[type[Model]]:
    """Return ``model`` and its proxies, which Django deletes under their own."""
    # Not apps.get_models(): a model is registered while the apps still load.
    found = [model]
    for subclass in model.__subclasses__():
        if subclass._meta.proxy:
            found.extend(_model_and_proxies(subclass))
    return found


def _models_reached_by_foreign_keys(
    model: type[Model], path: str
) -> list[tuple[str, type[Model]]]:
    """List the models the leading foreign keys of ``path`` reach from ``model``.

    Each is paired with the lookup, from ``model``, that reaches it.
    """
    names: list[str] = []
    reached: list[tuple[str, type[Model]]] = []
    for link in path_links(model, path):
        related_model = link.relation.related_model
        if not isinstance(link.relation, ForeignKey) or not isinstance(
            related_model, type
        ):
            break
        names.append(link.name)
        reached.append((LOOKUP_SEP.join(names), related_model))
    return reached


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
        # The lookups a custom extractor reads through, declared by its model.
        self._dependencies: dict[type[Model], tuple[str, ...]] = {}

    def registered_models(self) -> list[type[Model]]:
        """List the registered models, in registration order."""
        return list(self._registrations)

    def is_registered(self, model: type[Model]) -> bool:
        """Tell whether ``model`` is registered with fields or an extractor."""
        return model in self._registrations

    def _require_unregistered(self, model: type[Model]) -> None:
        """Fail if ``model`` is already registered with fields or an extractor.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
        """
        if self.is_registered(model):
            message = f"{model.__name__} is already registered"
            raise AlreadyRegistered(message)

    def _require_registered(self, model: type[Model]) -> None:
        """Fail unless ``model`` is registered with fields or an extractor.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        if not self.is_registered(model):
            message = f"{model.__name__} is not registered"
            raise NotRegistered(message)

    def new_extractor(self, model: type[Model]) -> BaseExtractor[Any]:
        """Build a fresh instance of the extractor ``model`` is registered with.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        self._require_registered(model)
        return self._registrations[model]()

    def register(  # noqa: PLR0913  # one keyword per registration option
        self,
        model: type[Model],
        *,
        fields: FieldNames | None = None,
        title_field: str | None = None,
        exclude: FieldNames = (),
        follow: FieldNames = (),
        language_field: str | None = None,
        language: str | None = None,
        url_field: str | None = None,
        permissions: PermissionNames = (),
    ) -> None:
        """Register ``model`` with the fields to extract.

        Without ``fields``, the model's text fields are extracted, except
        those named in ``exclude``.
        ``title_field`` names the field whose value is the document title.
        The text of the relations named in ``follow`` comes after the fields.
        ``language_field`` names the field whose value is the document
        language: an own field, or a lookup path such as ``page__language``.
        ``language`` gives every document of the model that language.
        ``url_field`` names the field whose stripped value is the document
        url: an own field, or a lookup path such as ``bookmark__link``.
        ``permissions`` are given to every document of the model.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: ``fields``, ``exclude``, ``follow`` or
                ``permissions`` is not a list or a tuple; no field is
                declared, a field is declared or excluded twice, or a field
                (``title_field``, ``language_field``, ``url_field`` and
                ``exclude`` included) is not one of the model's or is a
                relation;
                ``exclude`` is combined with ``fields``, or ``language`` with
                ``language_field``; ``language`` is blank or not a string;
                without ``fields`` and ``follow``, the model has no text
                field, or none left once ``exclude`` is applied; a name in
                ``follow`` is not one of the model's relation accessors, is
                given twice, or leads to a model with no text field; a name
                in ``exclude`` is a lookup path; or relations are followed
                while models are loading.
        """
        self._require_unregistered(model)
        _require_single_language_source(language, language_field)
        _require_text_language(language)
        _require_field_names(model, exclude, "exclude")
        _require_field_names(model, follow, "follow")
        _require_permission_names(permissions)
        _require_fields_or_exclude(model, fields, exclude)
        read_language_field = _language_field(model, language_field, language)
        if fields is None:
            _require_own_fields(model, exclude)
            _require_distinct_content_fields(model, exclude, "excluded")
            metadata_fields = (read_language_field, url_field)
            fields = _guessed_fields(model, exclude, follow, metadata_fields)
        else:
            _require_content_fields(model, fields)
        for single_field in (title_field, read_language_field, url_field):
            if single_field is not None:
                _require_content_field(model, single_field)
        if follow:
            _require_models_ready(model, "follow relations")
            _require_followable_relations(model, follow)
        declared = tuple(fields)
        followed = tuple(follow)
        granted = tuple(permissions)

        def build_extractor() -> DeclaredFieldsExtractor:
            return DeclaredFieldsExtractor(
                declared,
                title_field,
                followed,
                read_language_field,
                language,
                url_field,
                granted,
            )

        self._add(model, build_extractor)

    def register_extractor(
        self, model: type[M], *, depends_on: FieldNames = ()
    ) -> Callable[[type[BaseExtractor[M]]], type[BaseExtractor[Any]]]:
        """Register the decorated extractor class as the one of ``model``.

        ``depends_on`` names the foreign keys, reverse foreign keys, or lookup
        paths through foreign keys, whose saves change the documents of
        ``model``.

        Raises:
            AlreadyRegistered: ``model`` is already registered.
            ImproperlyConfigured: the decorated class does not derive from
                ``BaseExtractor``, or does not implement ``extract``, or
                ``depends_on`` is not a list or a tuple.
        """

        def decorator(
            extractor_class: type[BaseExtractor[M]],
        ) -> type[BaseExtractor[M]]:
            _require_extractor_class(extractor_class)
            _require_field_names(model, depends_on, "depends_on")
            self._require_unregistered(model)
            self._add(model, extractor_class, tuple(depends_on))
            return extractor_class

        return decorator

    def unregister(self, model: type[Model]) -> None:
        """Forget ``model``.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        self._require_registered(model)
        senders = self._delete_senders(model)
        del self._registrations[model]
        del self._dependencies[model]
        # A sender another registered model still listens to keeps its listener.
        kept = {
            sender
            for registered in self._registrations
            for sender in self._delete_senders(registered)
        }
        for sender in senders:
            if sender not in kept:
                post_delete.disconnect(dispatch_uid=_delete_uid(sender), sender=sender)
                pre_delete.disconnect(dispatch_uid=_delete_uid(sender), sender=sender)

    def _delete_senders(self, model: type[Model]) -> list[type[Model]]:
        """List the senders whose deletions change the group of ``model``.

        They are ``model`` and its proxies, and the models of the reverse
        foreign keys ``model`` follows, the models it reads through foreign
        keys, one or a chain, and their proxies.
        """
        followed_models = [
            relation.related_model
            for relation in self.followed_reverse_relations(model)
        ] + [reached for _, reached in self.foreign_key_lookups(model)]
        return _model_and_proxies(model) + [
            sender
            for followed_model in followed_models
            # A foreign key may name a proxy: Django deletes the concrete model's
            # rows under the concrete model, or under any of its proxies.
            for sender in _model_and_proxies(concrete_model_of(followed_model))
        ]

    def followed_reverse_relations(self, model: type[Model]) -> list[ForeignObjectRel]:
        """List the reverse foreign keys ``model`` follows or depends on.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        return [
            relation
            for relation in [
                *self._followed_relations(model),
                *self._dependency_relations(model),
            ]
            if isinstance(relation, ForeignObjectRel) and not relation.many_to_many
        ]

    def foreign_key_lookups(self, model: type[Model]) -> list[tuple[str, type[Model]]]:
        """List the models ``model`` reads through foreign keys, one or a chain.

        Those are the models of the foreign keys it follows, and the models
        its lookup paths and dependencies reach through their leading foreign
        keys. Each is
        paired with the lookup, from ``model``, that reaches it.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        followed = [
            (relation.name, relation.related_model)
            for relation in self._followed_relations(model)
            if isinstance(relation, ForeignKey)
        ]
        read_through_paths = [
            reached
            for path in [*self._lookup_paths(model), *self._dependency_paths(model)]
            for reached in _models_reached_by_foreign_keys(model, path)
        ]
        # Paths sharing a prefix, or a followed foreign key, reach a model twice.
        return list(dict.fromkeys(followed + read_through_paths))

    def _dependency_paths(self, model: type[Model]) -> list[str]:
        """List the lookup paths through the dependencies ``model`` declares."""
        # A dependency ends on a relation: a field after it makes it a lookup path.
        return [
            f"{dependency}{LOOKUP_SEP}pk" for dependency in self._dependencies[model]
        ]

    def _dependency_relations(
        self, model: type[Model]
    ) -> "list[Field[Any, Any] | ForeignObjectRel]":
        """List the relations, forward or reverse, ``model`` names as dependencies.

        A dependency that is a lookup path through relations is not one.
        """
        dependencies = self._dependencies[model]
        # Relations are only read when something is depended on: models may
        # still be loading otherwise.
        if not dependencies:
            return []

        relations = relations_by_accessor(model)
        return [
            relations[dependency]
            for dependency in dependencies
            if dependency in relations
        ]

    def _lookup_paths(self, model: type[Model]) -> list[str]:
        """List the lookup paths ``model`` declares in fields and single_fields.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        extractor = self.new_extractor(model)
        if not isinstance(extractor, DeclaredFieldsExtractor):
            return []
        return [*extractor.fields, *extractor.single_fields]

    def _followed_relations(
        self, model: type[Model]
    ) -> "list[Field[Any, Any] | ForeignObjectRel]":
        """List the relations, forward or reverse, ``model`` follows.

        Raises:
            NotRegistered: ``model`` is not registered.
        """
        extractor = self.new_extractor(model)
        # Relations are only read when something is followed: models may still
        # be loading otherwise.
        if not isinstance(extractor, DeclaredFieldsExtractor) or not extractor.follow:
            return []

        relations = relations_by_accessor(model)
        return [relations[accessor] for accessor in extractor.follow]

    def _add(
        self,
        model: type[Model],
        factory: Callable[[], BaseExtractor[Any]],
        depends_on: tuple[str, ...] = (),
    ) -> None:
        """Register ``model`` and listen to its deletions, and its own only."""
        from django_model_rag.signals import (  # noqa: PLC0415  # signals imports this module
            remember_followers_before_delete,
            sync_deleted_instance,
        )

        self._registrations[model] = factory
        self._dependencies[model] = depends_on
        # A listener without sender would also stop Django from fast-deleting
        # the models that are not registered.
        for sender in self._delete_senders(model):
            pre_delete.connect(
                remember_followers_before_delete,
                sender=sender,
                dispatch_uid=_delete_uid(sender),
            )
            post_delete.connect(
                sync_deleted_instance, sender=sender, dispatch_uid=_delete_uid(sender)
            )


rag = Registry()
