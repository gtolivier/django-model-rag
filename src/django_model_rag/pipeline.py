"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from typing import Any

from django.core.exceptions import FieldDoesNotExist
from django.db.models import Field, ForeignObjectRel, Model, QuerySet
from django.db.models.constants import LOOKUP_SEP

from django_model_rag.documents import NormalizedDocument
from django_model_rag.extractors import (
    BaseExtractor,
    DeclaredFieldsExtractor,
    PathLink,
    path_links,
)
from django_model_rag.registry import rag, relations_by_accessor

# iterator() prefetches per chunk: this many instances share one query
_CHUNK_SIZE = 1000

# the extractor options that each name one field, possibly a lookup path
_SINGLE_FIELD_OPTIONS = ("title_field", "language_field", "url_field")


def _option(extractor: BaseExtractor[Any], name: str, default: Any) -> Any:
    """Return the registration option ``name`` of ``extractor``, else ``default``.

    Only the extractor built from a registration has options: an attribute of
    a custom extractor, whatever its name, shapes nothing.
    """
    if isinstance(extractor, DeclaredFieldsExtractor):
        return getattr(extractor, name)
    return default


def _followed(extractor: BaseExtractor[Any]) -> Sequence[str]:
    """Return the names of the relations ``extractor`` follows, if any."""
    followed: Sequence[str] = _option(extractor, "follow", ())
    return followed


def _declared_fields(extractor: BaseExtractor[Any]) -> Sequence[str]:
    """Return the fields ``extractor`` declares, lookup paths included, if any."""
    fields: Sequence[str] = _option(extractor, "fields", ())
    return fields


def _read_fields(extractor: BaseExtractor[Any]) -> list[str]:
    """Return the fields ``extractor`` reads, lookup paths included.

    The fields read are those it declares, its title field, its language field
    and its URL field, if any.
    """
    named: list[str | None] = [
        _option(extractor, option, None) for option in _SINGLE_FIELD_OPTIONS
    ]
    return [*_declared_fields(extractor), *(name for name in named if name)]


def _lookup_paths(extractor: BaseExtractor[Any]) -> list[str]:
    """Return, once each, the lookup paths among the fields ``extractor`` reads."""
    return [
        name for name in dict.fromkeys(_read_fields(extractor)) if LOOKUP_SEP in name
    ]


# quoted: Django's Field is generic for the type checker only, and before
# Python 3.14 an annotation is evaluated when the function is defined
def _is_selected(relation: "Field[Any, Any] | ForeignObjectRel") -> bool:
    """Tell whether ``relation`` leads to a single object, read in the same query.

    These are the foreign keys and the one-to-one relations, forward or reverse.
    """
    if isinstance(relation, ForeignObjectRel):
        return bool(relation.one_to_one)
    # a generic foreign key leads to a single object too, but has no column
    # select_related() could join on
    return bool(relation.concrete and (relation.many_to_one or relation.one_to_one))


def _is_prefetched(relation: "Field[Any, Any] | ForeignObjectRel") -> bool:
    """Tell whether ``relation`` leads to many objects, read in one more query.

    These are the reverse foreign keys, the many-to-many relations, forward or
    reverse, and the generic relations.
    """
    return bool(relation.one_to_many or relation.many_to_many)


def _sorted_relations(
    model: type[Model], followed: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Sort the relations ``followed`` from ``model`` by how they are read.

    Returns:
        The names to give select_related(), then those to give
        prefetch_related().
    """
    relations = relations_by_accessor(model)
    selected: list[str] = []
    prefetched: list[str] = []
    for accessor in followed:
        relation = relations.get(accessor)
        if relation is None:
            continue
        if _is_selected(relation):
            # select_related() names a reverse relation by its query name,
            # which related_query_name may set apart from its accessor
            selected.append(relation.name)
        elif _is_prefetched(relation):
            prefetched.append(accessor)
    return selected, prefetched


def _selected_run(model: type[Model], path: str) -> list[PathLink]:
    """Return the single-object relations ``path`` starts with."""
    run: list[PathLink] = []
    # a ``fields`` attribute of a custom extractor need not be paths
    with suppress(FieldDoesNotExist):
        for link in path_links(model, path):
            if not _is_selected(link.relation):
                break
            run.append(link)
    return run


def _query_path(run: Sequence[PathLink]) -> str:
    """Join the links of ``run`` into the lookup select_related() names it by."""
    return LOOKUP_SEP.join(link.query_name for link in run)


def _selected_path_prefixes(
    model: type[Model], extractor: BaseExtractor[Any]
) -> list[str]:
    """Return the run of single-object relations each lookup path starts with."""
    runs = (_selected_run(model, path) for path in _lookup_paths(extractor))
    return [_query_path(run) for run in runs if run]


def _read_by_prefix(
    model: type[Model], extractor: BaseExtractor[Any]
) -> dict[str, tuple[type[Model], set[str]]]:
    """Map each selected relation prefix of the lookup paths to what is read there.

    The value is the related model, and the names of its fields the paths read.
    """
    followed = set(_followed(extractor))
    read: dict[str, tuple[type[Model], set[str]]] = {}
    for path in _lookup_paths(extractor):
        run = _selected_run(model, path)
        # a followed relation is read in full
        if run and run[0].accessor in followed:
            continue
        names = path.split(LOOKUP_SEP)
        for depth, link in enumerate(run, start=1):
            owner = link.relation.related_model
            # for the type checker only: a selected relation leads to a model
            if owner is None:
                break
            prefix = _query_path(run[:depth])
            read.setdefault(prefix, (owner, set()))[1].add(names[depth])
    return read


def _unread_related_columns(
    model: type[Model], extractor: BaseExtractor[Any]
) -> list[str]:
    """Return the lookup names of the related columns no lookup path reads."""
    return [
        f"{prefix}{LOOKUP_SEP}{name}"
        for prefix, (owner, read) in _read_by_prefix(model, extractor).items()
        for name in _unread_field_names(owner, read)
    ]


def _unread_field_names(model: type[Model], read: set[str]) -> list[str]:
    """Return the names of ``model``'s columns that ``read`` does not name.

    The primary key is always read. A foreign key may be named by its field
    or by its column.
    """
    return [
        field.name
        for field in model._meta.concrete_fields
        if not field.primary_key and not {field.name, field.attname} & read
    ]


def _reads_only_named_columns(
    model: type[Model], extractor: BaseExtractor[Any]
) -> bool:
    """Tell whether the own columns ``extractor`` reads are all named by it.

    Only the fields it declares name them, and get_absolute_url() may read any
    column, when no URL field replaces it.
    """
    return bool(_declared_fields(extractor)) and bool(
        _option(extractor, "url_field", None) or not hasattr(model, "get_absolute_url")
    )


def _unread_columns(model: type[Model], extractor: BaseExtractor[Any]) -> list[str]:
    """Return the names of the own columns ``extractor`` never reads.

    It reads the first link of each lookup path it declares, and each relation
    it follows: they stay, with the own fields it names.
    """
    read = {
        name.split(LOOKUP_SEP)[0]
        for name in (*_read_fields(extractor), *_followed(extractor))
    }
    return _unread_field_names(model, read)


def _instances(model: type[Model], extractor: BaseExtractor[Any]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order.

    The followed foreign keys and one-to-one relations, and the run of them
    each lookup path starts with, come with each instance, in the same query.
    The followed reverse foreign keys, many-to-many relations, forward or
    reverse, and generic relations come in one more query each for all the
    instances.
    """
    queryset = model._default_manager.order_by("pk")
    if _reads_only_named_columns(model, extractor):
        queryset = queryset.defer(*_unread_columns(model, extractor))
    selected, prefetched = _sorted_relations(model, _followed(extractor))
    selected.extend(_selected_path_prefixes(model, extractor))
    # never select_related() without a field: it would follow every non-null
    # foreign key, followed or not
    if selected:
        queryset = queryset.select_related(*selected)
        if unread := _unread_related_columns(model, extractor):
            queryset = queryset.defer(*unread)
    if prefetched:
        queryset = queryset.prefetch_related(*prefetched)
    hooked = _checked_queryset(extractor.get_queryset(queryset), extractor)
    return hooked.iterator(chunk_size=_CHUNK_SIZE)


def _checked_queryset(hooked: object, extractor: BaseExtractor[Any]) -> QuerySet[Model]:
    """Return what ``extractor``'s get_queryset() hooked, failing on a non-queryset."""
    if not isinstance(hooked, QuerySet):
        raise TypeError(
            f"{type(extractor).__name__}.get_queryset() must return a QuerySet, "
            f"not a {type(hooked).__name__}"
        )
    return hooked


def _wrong_extraction(extractor: BaseExtractor[Any], returned: str) -> TypeError:
    """Build the error for an ``extractor`` whose extract() ``returned`` no document."""
    return TypeError(
        f"{type(extractor).__name__}.extract() returned {returned}, "
        "not a NormalizedDocument"
    )


def _instance_documents(
    instance: Model, extractor: BaseExtractor[Any]
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``instance`` with ``extractor``."""
    extracted = extractor.extract(instance)
    if extracted is None:
        return
    if isinstance(extracted, NormalizedDocument):
        yield extracted
    elif isinstance(extracted, str):
        raise _wrong_extraction(extractor, "a string")
    elif not isinstance(extracted, Iterable):
        raise _wrong_extraction(extractor, f"a {type(extracted).__name__}")
    else:
        yield from _checked_documents(extracted, extractor)


def _checked_documents(
    extracted: Iterable[object], extractor: BaseExtractor[Any]
) -> Iterator[NormalizedDocument]:
    """Yield the items ``extractor`` extracted, failing on the first non-document."""
    for item in extracted:
        if not isinstance(item, NormalizedDocument):
            raise _wrong_extraction(extractor, f"a {type(item).__name__}")
        yield item


def _model_documents(
    model: type[Model], extractor: BaseExtractor[Any]
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``model``'s instances, in primary key order."""
    for instance in _instances(model, extractor):
        yield from _instance_documents(instance, extractor)


def _extractors_to_run(
    models: Sequence[type[Model]] | None,
) -> list[tuple[type[Model], BaseExtractor[Any]]]:
    """Pair each model to run with its extractor.

    The models to run are ``models``, in their order, or every registered model.

    Raises:
        NotRegistered: one of ``models`` is not registered.
    """
    if models is None:
        models = rag.registered_models()
    return [(model, rag.new_extractor(model)) for model in models]


class SyncPipeline:
    """Turn registered models into normalized documents."""

    def run(
        self, models: Sequence[type[Model]] | None = None
    ) -> list[NormalizedDocument]:
        """Produce the documents of the registered models.

        Only the given ``models`` are run, in their order, or every registered
        model by default.

        Raises:
            NotRegistered: one of ``models`` is not registered.
        """
        documents: list[NormalizedDocument] = []
        for model, extractor in _extractors_to_run(models):
            documents.extend(_model_documents(model, extractor))
        return documents

    def run_instance(self, instance: Model) -> list[NormalizedDocument]:
        """Produce the documents of ``instance`` only.

        Raises:
            NotRegistered: the model of ``instance`` is not registered.
        """
        extractor = rag.new_extractor(type(instance))
        return list(_instance_documents(instance, extractor))
