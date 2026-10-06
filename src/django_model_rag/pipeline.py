"""The pipeline that turns registered models into normalized documents for an output."""

from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import Any

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


def _instances(model: type[Model], extractor: BaseExtractor[Any]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order.

    ``extractor``'s get_queryset() decides which are loaded, and how.
    """
    hooked = _hooked_queryset(
        model._default_manager.order_by(_DOCUMENT_ORDER), extractor, model
    )
    return hooked.order_by(_DOCUMENT_ORDER).iterator(chunk_size=_CHUNK_SIZE)


def _current_keys(model: type[Model], extractor: BaseExtractor[Any]) -> set[str]:
    """Return the source keys of ``model``'s instances ``extractor`` keeps now."""
    kept_pks = _kept_queryset(model, extractor).values_list("pk", flat=True)
    return {
        model_source_key(model, pk) for pk in kept_pks.iterator(chunk_size=_CHUNK_SIZE)
    }


def _kept_queryset(
    model: type[Model], extractor: BaseExtractor[Any]
) -> QuerySet[Model]:
    """Return ``model``'s instances that ``extractor``'s get_queryset() keeps.

    Raises:
        TypeError: get_queryset() did not return a QuerySet of ``model``'s instances.
    """
    return _hooked_queryset(model._default_manager.all(), extractor, model)


def _hooked_queryset(
    queryset: QuerySet[Model], extractor: BaseExtractor[Any], model: type[Model]
) -> QuerySet[Model]:
    """Pass ``queryset`` of ``model`` through ``extractor``'s get_queryset().

    Raises:
        TypeError: get_queryset() did not return a QuerySet of ``model``'s instances.
    """
    return _checked_queryset(extractor.get_queryset(queryset), extractor, model)


def _is_kept_by_hook(instance: Model, extractor: BaseExtractor[Any]) -> bool:
    """Tell whether ``extractor``'s get_queryset() keeps ``instance``.

    Raises:
        TypeError: get_queryset() did not return a QuerySet of the model's instances.
    """
    return _kept_queryset(type(instance), extractor).filter(pk=instance.pk).exists()


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


def _distinct(instances: Iterable[Model]) -> Iterator[Model]:
    """Yield ``instances``, skipping the ones whose source key was already yielded."""
    seen_keys: set[str] = set()
    for instance in instances:
        source_key = _source_key(instance)
        if source_key not in seen_keys:
            seen_keys.add(source_key)
            yield instance


def _groups(
    instances: Iterable[Model], extractor: BaseExtractor[Any]
) -> dict[str, list[NormalizedDocument]]:
    """Group the documents of ``instances`` by source key, each instance once.

    An instance without documents gets an empty group.
    """
    return {
        _source_key(instance): list(_own_documents(instance, extractor))
        for instance in _distinct(instances)
    }


def _hand_over(
    instances: Iterable[Model], extractor: BaseExtractor[Any], output: DocumentOutput
) -> set[str]:
    """Hand the documents of ``instances`` to ``output`` at once, grouped by source key.

    Instances without documents are handed over as empty groups. Returns the
    source keys handed over with documents.
    """
    groups = _groups(instances, extractor)
    if groups:
        output.replace(groups)
    return {source_key for source_key, documents in groups.items() if documents}


def _keys_to_keep(
    handed_keys: set[str], read_keys: set[str], current_keys: set[str]
) -> set[str]:
    """Return the source keys a model's prune keeps, once its run is over.

    The prune keeps the ``current_keys`` but those read without documents. An
    instance deleted during the run was handed over, but is gone. An
    instance created during the run was not read: its own signal handed its
    documents over, which the prune must not delete.

    Every handed key was read, so only the keys read without documents need a
    set of their own, not two the size of the table.
    """
    return current_keys - (read_keys - handed_keys)


class SyncPipeline:
    """Turn registered models into normalized documents, handed to an output."""

    def __init__(self, output: DocumentOutput) -> None:
        """Hand the documents to ``output``."""
        self._output = output

    def run(self, models: Sequence[type[Model]] | None = None) -> None:
        """Hand the documents of the registered models to the output.

        Only the given ``models`` are run, in their order, or every registered
        model by default. Each model is then pruned down to the source keys of
        the instances its extractor keeps once the model is run, less those
        read without producing documents.

        Raises:
            NotRegistered: one of ``models`` is not registered.
        """
        for model, extractor in _extractors_to_run(models):
            self._run_model(model, extractor)

    def _run_model(self, model: type[Model], extractor: BaseExtractor[Any]) -> None:
        """Hand ``model``'s documents over chunk by chunk, then prune the model."""
        handed_keys: set[str] = set()
        read_keys: set[str] = set()
        instances = _instances(model, extractor)
        while chunk := list(islice(instances, _CHUNK_SIZE)):
            read_keys.update(_source_key(instance) for instance in chunk)
            handed_keys |= _hand_over(chunk, extractor, self._output)
        kept_keys = _keys_to_keep(
            handed_keys, read_keys, _current_keys(model, extractor)
        )
        self._output.prune(model._meta.label_lower, kept_keys)

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
        documents = (
            list(_own_documents(instance, extractor))
            if _is_kept_by_hook(instance, extractor)
            else []
        )
        self._output.replace({_source_key(instance): documents})
