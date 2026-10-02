"""The pipeline that turns registered models into normalized documents."""

from collections.abc import Iterator

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument
from django_model_rag.registry import Declaration, rag

_FIELD_SEPARATOR = "\n\n"


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
    return NormalizedDocument(
        text=text,
        source_app_label=instance._meta.app_label,  # Django's public meta API
        source_model=instance._meta.model_name or "",  # Django's public meta API
        source_pk=instance.pk,
        title=_document_title(instance, declaration, field_texts),
    )


def _model_documents(
    model: type[Model], declaration: Declaration
) -> Iterator[NormalizedDocument]:
    """Build the documents of ``model``'s instances, in primary key order."""
    for instance in model._default_manager.order_by("pk").iterator():
        document = _document(instance, declaration)
        if document is not None:
            yield document


class SyncPipeline:
    """Turn registered models into normalized documents."""

    def run(self) -> list[NormalizedDocument]:
        """Produce the documents of every registered model."""
        return [
            document
            for model, declaration in rag.declarations()
            for document in _model_documents(model, declaration)
        ]
