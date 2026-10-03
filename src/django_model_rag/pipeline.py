"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument
from django_model_rag.extractors import BaseExtractor
from django_model_rag.registry import rag


def _followed_foreign_keys(
    model: type[Model], extractor: BaseExtractor[Any]
) -> list[str]:
    """List the relations ``extractor`` follows that are foreign keys of ``model``."""
    followed = getattr(extractor, "follow", ())
    # a followed name may be a reverse accessor, which is no field name:
    # look among the concrete fields instead of calling get_field()
    return [
        field.name
        for field in model._meta.concrete_fields
        if field.many_to_one and field.name in followed
    ]


def _instances(model: type[Model], extractor: BaseExtractor[Any]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order.

    The followed foreign keys come with each instance, in the same query.
    """
    queryset = model._default_manager.order_by("pk")
    # select_related() without a field is deprecated
    if foreign_keys := _followed_foreign_keys(model, extractor):
        queryset = queryset.select_related(*foreign_keys)
    return queryset.iterator()


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
