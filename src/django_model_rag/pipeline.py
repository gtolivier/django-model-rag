"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from typing import Any

from django.db.models import Model, QuerySet
from django.db.models.query import ModelIterable

from django_model_rag.documents import NormalizedDocument, build_source_key
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

    ``extractor``'s get_queryset() shapes how they are loaded.
    """
    queryset = model._default_manager.order_by(_DOCUMENT_ORDER)
    hooked = _checked_queryset(extractor.get_queryset(queryset), extractor, model)
    return hooked.order_by(_DOCUMENT_ORDER).iterator(chunk_size=_CHUNK_SIZE)


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


def _groups(
    instances: Iterable[Model], extractor: BaseExtractor[Any]
) -> dict[str, list[NormalizedDocument]]:
    """Group the documents of ``instances`` by source key."""
    groups: dict[str, list[NormalizedDocument]] = {}
    for instance in instances:
        for document in _instance_documents(instance, extractor):
            groups.setdefault(document.source_key, []).append(document)
    return groups


def _hand_over(
    instances: Iterable[Model], extractor: BaseExtractor[Any], output: DocumentOutput
) -> set[str]:
    """Hand the documents of ``instances`` to ``output`` at once, grouped by source key.

    Returns the source keys handed over.
    """
    groups = _groups(instances, extractor)
    if groups:
        output.replace(groups)
    return set(groups)


class SyncPipeline:
    """Turn registered models into normalized documents, handed to an output."""

    def __init__(self, output: DocumentOutput) -> None:
        """Hand the documents to ``output``."""
        self._output = output

    def run(self, models: Sequence[type[Model]] | None = None) -> None:
        """Produce the documents of the registered models.

        Only the given ``models`` are run, in their order, or every registered
        model by default.

        Raises:
            NotRegistered: one of ``models`` is not registered.
        """
        for model, extractor in _extractors_to_run(models):
            kept_keys: set[str] = set()
            instances = _instances(model, extractor)
            while chunk := list(islice(instances, _CHUNK_SIZE)):
                kept_keys |= _hand_over(chunk, extractor, self._output)
            self._output.prune(model._meta.label_lower, kept_keys)

    def run_instance(self, instance: Model) -> None:
        """Produce the documents of ``instance`` only.

        Raises:
            NotRegistered: the model of ``instance`` is not registered.
        """
        extractor = rag.new_extractor(type(instance))
        groups = _groups([instance], extractor)
        # an empty group still replaces what the output holds for the instance
        groups.setdefault(build_source_key(instance._meta.label_lower, instance.pk), [])
        self._output.replace(groups)
