"""The pipeline that turns registered models into normalized documents for an output."""

from collections.abc import Iterable, Iterator, Sequence
from itertools import groupby, islice
from typing import Any, TypeVar

from django.db.models import Model, QuerySet
from django.db.models.query import ModelIterable

from django_model_rag.documents import NormalizedDocument, model_source_key
from django_model_rag.extractors import BaseExtractor
from django_model_rag.output import DocumentOutput
from django_model_rag.registry import rag

# iterator() prefetches per chunk: this many instances share one query, and
# their documents one replace() call
_CHUNK_SIZE = 1000
# the order of a model's documents, whatever its extractor's get_queryset() asks
_DOCUMENT_ORDER = "pk"

_Item = TypeVar("_Item")


def _instances(model: type[Model], extractor: BaseExtractor[Any]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order.

    ``extractor``'s get_queryset() decides which are loaded, and how.
    """
    hooked = _hooked_queryset(
        model._default_manager.order_by(_DOCUMENT_ORDER), extractor, model
    )
    return _in_document_order(hooked)


def _in_document_order(queryset: QuerySet[Model]) -> Iterator[Model]:
    """Iterate over the instances of ``queryset``, in primary key order."""
    return queryset.order_by(_DOCUMENT_ORDER).iterator(chunk_size=_CHUNK_SIZE)


def _pks_in_document_order(queryset: QuerySet[Model]) -> Iterator[Any]:
    """Iterate over the primary keys of ``queryset``'s instances, in their order."""
    pks = queryset.order_by(_DOCUMENT_ORDER).values_list("pk", flat=True)
    return _skipping_adjacent_repeats(pks.iterator(chunk_size=_CHUNK_SIZE))


def _skipping_adjacent_repeats(rows: Iterable[_Item]) -> Iterator[_Item]:
    """Yield ``rows``, skipping a repeat of the previous row.

    A join can repeat an instance's row. Only adjacent repeats are skipped:
    ``rows`` must come in primary key order, which puts all the rows of an
    instance next to each other, so each is yielded once.
    """
    return (row for row, _ in groupby(rows))


def _current_keys(model: type[Model], extractor: BaseExtractor[Any]) -> set[str]:
    """Return the source keys of ``model``'s instances ``extractor`` keeps now."""
    kept_pks = _kept_queryset(model, extractor).values_list("pk", flat=True)
    return {
        model_source_key(model, pk) for pk in kept_pks.iterator(chunk_size=_CHUNK_SIZE)
    }


def _kept_queryset(
    model: type[Model], extractor: BaseExtractor[Any], using: str | None = None
) -> QuerySet[Model]:
    """Return ``model``'s instances that ``extractor``'s get_queryset() keeps.

    They are read from the database alias ``using``, or the one Django's
    routers pick by default.

    Raises:
        TypeError: get_queryset() did not return a QuerySet of ``model``'s instances.
    """
    return _hooked_queryset(model._default_manager.using(using), extractor, model)


def _hooked_queryset(
    queryset: QuerySet[Model], extractor: BaseExtractor[Any], model: type[Model]
) -> QuerySet[Model]:
    """Pass ``queryset`` of ``model`` through ``extractor``'s get_queryset().

    Raises:
        TypeError: get_queryset() did not return a QuerySet of ``model``'s instances.
    """
    return _checked_queryset(extractor.get_queryset(queryset), extractor, model)


def _reloaded_by_hook(pks: Iterable[Any], kept: QuerySet[Model]) -> dict[Any, Model]:
    """Reload the instances of ``pks`` from ``kept``, as get_queryset() hooked it.

    What get_queryset() adds to them (an annotation...) then reaches extract().
    The result maps the primary key of each instance it keeps to its reload.
    """
    # streamed: a join in get_queryset() can multiply the rows to cache
    reloads = kept.filter(pk__in=pks).iterator(chunk_size=_CHUNK_SIZE)
    return {instance.pk: instance for instance in reloads}


def _checked_queryset(
    hooked: object, extractor: BaseExtractor[Any], model: type[Model]
) -> QuerySet[Model]:
    """Return what ``extractor``'s get_queryset() hooked.

    Raises:
        TypeError: it is not a QuerySet of ``model``'s instances.
    """
    if not isinstance(hooked, QuerySet):
        raise _wrong_queryset(extractor, "a QuerySet", f"a {type(hooked).__name__}")
    # No public API tells a values() queryset from one of instances.
    if not issubclass(hooked._iterable_class, ModelIterable):
        raise _wrong_queryset(extractor, "a QuerySet of model instances", "of values")
    if hooked.model is not model:
        raise _wrong_queryset(
            extractor,
            f"a QuerySet of {model.__name__}",
            f"of {hooked.model.__name__}",
        )
    return hooked


def _wrong_queryset(
    extractor: BaseExtractor[Any], expected: str, returned: str
) -> TypeError:
    """Build the error for an ``extractor``'s get_queryset() not ``expected``."""
    return TypeError(
        f"{type(extractor).__name__}.get_queryset() must return {expected}, "
        f"not {returned}"
    )


def _wrong_extraction(extractor: BaseExtractor[Any], returned: str) -> TypeError:
    """Build the error for an ``extractor`` whose extract() ``returned`` no document."""
    return TypeError(
        f"{type(extractor).__name__}.extract() returned {returned}, "
        "not a NormalizedDocument"
    )


def _foreign_source(extractor: BaseExtractor[Any], source_key: str) -> TypeError:
    """Build the error for an ``extractor`` whose document is not of ``source_key``."""
    return TypeError(
        f"{type(extractor).__name__}.extract() returned a document "
        f"whose source is not {source_key}"
    )


def _source_key(instance: Model) -> str:
    """Return the source key of ``instance``'s documents."""
    return model_source_key(instance._meta.model, instance.pk)


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


def _own_documents(
    instance: Model, extractor: BaseExtractor[Any]
) -> Iterator[NormalizedDocument]:
    """Yield the documents of ``instance``, failing on the first of another source."""
    source_key = _source_key(instance)
    for document in _instance_documents(instance, extractor):
        if document.source_key != source_key:
            raise _foreign_source(extractor, source_key)
        yield document


def _reloaded_documents(
    reloaded: Model | None, extractor: BaseExtractor[Any]
) -> list[NormalizedDocument]:
    """Return the documents of an instance as get_queryset() ``reloaded`` it.

    There are none if get_queryset() filtered the instance out (``reloaded``
    is then None): it is not extracted.

    Raises:
        TypeError: ``extractor``'s extract() returned a document of another source.
    """
    if reloaded is None:
        return []
    return list(_own_documents(reloaded, extractor))


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


def _keyed_skipping_adjacent_repeats(
    instances: Iterable[Model],
) -> Iterator[tuple[str, Model]]:
    """Pair ``instances`` with their source key, skipping a repeat of the previous key.

    ``instances`` must come in primary key order (see _skipping_adjacent_repeats).
    """
    keyed = ((_source_key(instance), instance) for instance in instances)
    return _skipping_adjacent_repeats(keyed)


def _chunks(items: Iterable[_Item]) -> Iterator[list[_Item]]:
    """Cut ``items`` into lists of ``_CHUNK_SIZE``, the last one possibly shorter.

    Only one chunk is held at a time.
    """
    remaining = iter(items)
    while chunk := list(islice(remaining, _CHUNK_SIZE)):
        yield chunk


def _groups(
    keyed: Iterable[tuple[str, Model]], extractor: BaseExtractor[Any]
) -> dict[str, list[NormalizedDocument]]:
    """Group the documents of the keyed instances by source key.

    An instance without documents gets an empty group.
    """
    return {
        source_key: list(_own_documents(instance, extractor))
        for source_key, instance in keyed
    }


def _reloaded_groups(
    pks: Sequence[Any], kept: QuerySet[Model], extractor: BaseExtractor[Any]
) -> dict[str, list[NormalizedDocument]]:
    """Group by source key the documents of the instances of ``pks``, reloaded.

    They are reloaded from ``kept``, the queryset ``extractor``'s get_queryset()
    hooked; an instance it filters out, or without documents, gets an empty group.

    Raises:
        TypeError: extract() returned a document of another source.
    """
    reloaded = _reloaded_by_hook(pks, kept)
    return {
        model_source_key(kept.model, pk): _reloaded_documents(
            reloaded.get(pk), extractor
        )
        for pk in pks
    }


class SyncPipeline:
    """Turn registered models into normalized documents, handed to an output."""

    def __init__(self, output: DocumentOutput) -> None:
        """Hand the documents to ``output``."""
        self._output = output

    def run(self, models: Sequence[type[Model]] | None = None) -> None:
        """Hand the documents of the registered models to the output.

        Only the given ``models`` are run, in their order, or every registered
        model by default. Each model is then pruned down to the source keys of
        the instances its extractor keeps once the model is run, those read
        without producing documents included: their empty group already
        removed their documents.

        Raises:
            NotRegistered: one of ``models`` is not registered.
        """
        for model, extractor in _extractors_to_run(models):
            self._run_model(model, extractor)

    def _run_model(self, model: type[Model], extractor: BaseExtractor[Any]) -> None:
        """Hand ``model``'s documents over chunk by chunk, then prune the model.

        Each chunk is handed over at once, grouped by source key; its instances
        without documents are handed over as empty groups.
        """
        keyed = _keyed_skipping_adjacent_repeats(_instances(model, extractor))
        for chunk in _chunks(keyed):
            self._output.replace(_groups(chunk, extractor))
        self._output.prune(model._meta.label_lower, _current_keys(model, extractor))

    def run_queryset(self, queryset: QuerySet[Any]) -> None:
        """Hand the documents of the instances of ``queryset`` only to the output.

        Raises:
            NotRegistered: the model of ``queryset`` is not registered, even
                when ``queryset`` is empty.
            TypeError: its extractor's get_queryset() did not return a QuerySet
                of the model's instances, even when ``queryset`` is empty; or
                ``queryset`` is sliced.
        """
        # No public API tells a sliced queryset from one that is not.
        if queryset.query.is_sliced:
            msg = "run_queryset() needs a queryset that is not sliced"
            raise TypeError(msg)
        extractor = rag.new_extractor(queryset.model)
        # hooked before the first chunk, a broken get_queryset() fails even
        # when there is nothing to run
        kept = _kept_queryset(queryset.model, extractor, using=queryset.db)
        for pks in _chunks(_pks_in_document_order(queryset)):
            self._output.replace(_reloaded_groups(pks, kept, extractor))

    def run_instance(self, instance: Model) -> None:
        """Hand the documents of ``instance`` only to the output, as one group.

        The group is empty when its extractor's get_queryset() filters
        ``instance`` out: it is then not extracted.

        Raises:
            NotRegistered: the model of ``instance`` is not registered.
            ValueError: ``instance`` has no primary key yet.
            TypeError: its extractor's get_queryset() did not return a QuerySet
                of the model's instances, or its extract() returned a document
                of another source.
        """
        extractor = rag.new_extractor(type(instance))
        if instance.pk is None:
            msg = "run_instance() needs a saved instance: its primary key is None"
            raise ValueError(msg)
        kept = _kept_queryset(type(instance), extractor)
        self._output.replace(_reloaded_groups([instance.pk], kept, extractor))
