"""The pipeline that turns registered models into normalized documents."""

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument
from django_model_rag.registry import rag

_FIELD_SEPARATOR = "\n\n"


def _field_text(instance: Model, name: str) -> str:
    """Read the field ``name`` of ``instance`` as text."""
    value = getattr(instance, name)
    return "" if value is None else str(value)


def _is_blank(text: str) -> bool:
    """Tell whether ``text`` holds nothing but whitespace."""
    return not text.strip()


def _document_text(field_texts: list[str]) -> str:
    """Join the non-blank ``field_texts``, stripped, in order."""
    return _FIELD_SEPARATOR.join(
        text.strip() for text in field_texts if not _is_blank(text)
    )


def _document_title(field_texts: list[str]) -> str:
    """Take the first of the ``field_texts``, or nothing when it is blank."""
    return field_texts[0].strip()


def _document(instance: Model, fields: list[str]) -> NormalizedDocument | None:
    """Build the document of ``instance``, or nothing when its ``fields`` are blank."""
    field_texts = [_field_text(instance, name) for name in fields]
    text = _document_text(field_texts)
    if not text:
        return None
    return NormalizedDocument(
        text=text,
        source_app_label=instance._meta.app_label,  # Django's public meta API
        source_model=instance._meta.model_name or "",  # Django's public meta API
        source_pk=instance.pk,
        title=_document_title(field_texts),
    )


class SyncPipeline:
    """Turn registered models into normalized documents."""

    def run(self) -> list[NormalizedDocument]:
        """Produce the documents of every registered model."""
        return [
            document
            for model, fields in rag.declarations()
            for instance in model._default_manager.order_by("pk").iterator()
            if (document := _document(instance, fields)) is not None
        ]
