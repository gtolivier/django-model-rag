"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from django.db.models import Field, ForeignObjectRel, Model

from django_model_rag.documents import NormalizedDocument
from django_model_rag.extractors import BaseExtractor
from django_model_rag.registry import rag, relations_by_accessor

# iterator() prefetches per chunk: this many instances share one query
_CHUNK_SIZE = 1000


def _followed(extractor: BaseExtractor[Any]) -> Sequence[str]:
    """Return the names of the relations ``extractor`` follows, if any."""
    followed: Sequence[str] = getattr(extractor, "follow", ())
    return followed


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


def _instances(model: type[Model], extractor: BaseExtractor[Any]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order.

    The followed foreign keys and one-to-one relations come with each instance,
    in the same query, and the followed reverse foreign keys, many-to-many
    relations, forward or reverse, and generic relations, in one more query
    each for all the instances.
    """
    queryset = model._default_manager.order_by("pk")
    selected, prefetched = _sorted_relations(model, _followed(extractor))
    # never select_related() without a field: it would follow every non-null
    # foreign key, followed or not
    if selected:
        queryset = queryset.select_related(*selected)
    if prefetched:
        queryset = queryset.prefetch_related(*prefetched)
    return queryset.iterator(chunk_size=_CHUNK_SIZE)


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
