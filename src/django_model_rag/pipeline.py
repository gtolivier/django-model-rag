"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Collection, Iterator
from typing import Any, TypeAlias

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument
from django_model_rag.extractors import BaseExtractor, document_from_instance
from django_model_rag.registry import Declaration, rag

_FIELD_SEPARATOR = "\n\n"

_Registration: TypeAlias = Declaration | BaseExtractor[Any]
"""How a registered model turns into documents: declared fields or an extractor."""


def _field_text(instance: Model, name: str) -> str:
    """Read the field ``name`` of ``instance`` as stripped text.

    A field with choices reads as its label.
    """
    value = getattr(instance, name)
    # Django adds get_<name>_display only to fields that have choices
    if getattr(instance._meta.get_field(name), "choices", None):
        value = getattr(instance, f"get_{name}_display")()
    return "" if value is None else str(value).strip()


def _document_text(field_texts: list[str]) -> str:
    """Join the non-empty ``field_texts``, in order."""
    return _FIELD_SEPARATOR.join(text for text in field_texts if text)


def _document_title(
    instance: Model, declaration: Declaration, field_texts: list[str]
) -> str:
    """Take the text of the declared title field, or the first of ``field_texts``.

    ``field_texts`` are the texts of the declared fields, in order: a title
    field among them is not read a second time.
    """
    title_field = declaration.title_field
    if title_field is None:
        return field_texts[0]
    if title_field in declaration.fields:
        return field_texts[declaration.fields.index(title_field)]
    return _field_text(instance, title_field)


def _document(instance: Model, declaration: Declaration) -> NormalizedDocument | None:
    """Build the document of ``instance``, or nothing when its fields are blank."""
    field_texts = [_field_text(instance, name) for name in declaration.fields]
    text = _document_text(field_texts)
    if not text:
        return None
    return document_from_instance(
        instance, text=text, title=_document_title(instance, declaration, field_texts)
    )


def _instances(model: type[Model]) -> Iterator[Model]:
    """Iterate over ``model``'s instances, in primary key order."""
    return model._default_manager.order_by("pk").iterator()


def _declared_instance_documents(
    instance: Model, declaration: Declaration
) -> Iterator[NormalizedDocument]:
    """Build the document of ``instance``, if its declared fields are not blank."""
    document = _document(instance, declaration)
    if document is not None:
        yield document


def _extracted_instance_documents(
    instance: Model, extractor: BaseExtractor[Any]
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``instance`` with ``extractor``."""
    extracted = extractor.extract(instance)
    if isinstance(extracted, NormalizedDocument):
        yield extracted
    elif isinstance(extracted, str):
        message = (
            f"{type(extractor).__name__}.extract() returned a string, "
            "not a NormalizedDocument"
        )
        raise TypeError(message)
    elif extracted is not None:
        yield from extracted


def _registrations() -> dict[type[Model], _Registration]:
    """Map each registered model to its declaration or its extractor."""
    return {**dict(rag.declarations()), **dict(rag.extractors())}


def _instance_documents(
    instance: Model, registration: _Registration
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``instance`` as its model's ``registration`` says."""
    if isinstance(registration, Declaration):
        return _declared_instance_documents(instance, registration)
    return _extracted_instance_documents(instance, registration)


def _model_documents(
    model: type[Model], registration: _Registration
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``model``'s instances, in primary key order."""
    for instance in _instances(model):
        yield from _instance_documents(instance, registration)


def _models_to_run(models: Collection[type[Model]] | None) -> list[type[Model]]:
    """Select ``models``, in their order, or every registered model.

    Raises:
        NotRegistered: one of ``models`` is not registered.
    """
    if models is None:
        return rag.models()
    for model in models:
        rag.require_registered(model)
    return list(models)


class SyncPipeline:
    """Turn registered models into normalized documents."""

    def run(
        self, models: Collection[type[Model]] | None = None
    ) -> list[NormalizedDocument]:
        """Produce the documents of the registered models.

        Only the given ``models`` are run, in their order, or every registered
        model by default.

        Raises:
            NotRegistered: one of ``models`` is not registered.
        """
        registrations = _registrations()
        documents: list[NormalizedDocument] = []
        for model in _models_to_run(models):
            documents.extend(_model_documents(model, registrations[model]))
        return documents

    def run_instance(self, instance: Model) -> list[NormalizedDocument]:
        """Produce the documents of ``instance`` only.

        Raises:
            NotRegistered: the model of ``instance`` is not registered.
        """
        rag.require_registered(type(instance))
        registration = _registrations()[type(instance)]
        return list(_instance_documents(instance, registration))
