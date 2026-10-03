"""The extractors: the base class of custom ones, and the one of declared fields."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument

M = TypeVar("M", bound=Model)


class BaseExtractor(ABC, Generic[M]):
    """Build the document(s) of an instance of a model."""

    @abstractmethod
    def extract(
        self, instance: M
    ) -> NormalizedDocument | Iterable[NormalizedDocument] | None:
        """Build the document(s) of ``instance``, or nothing to skip it."""

    def build_document(  # noqa: PLR0913  # one keyword per document field
        self,
        instance: M,
        *,
        text: str,
        title: str = "",
        url: str = "",
        language: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> NormalizedDocument:
        """Build a document with ``text``, its source taken from ``instance``."""
        return NormalizedDocument(
            text=text,
            source_app_label=instance._meta.app_label,  # Django's public meta API
            source_model=instance._meta.model_name or "",  # Django's public meta API
            source_pk=instance.pk,
            title=title,
            url=url,
            language=language,
            metadata=metadata or {},
        )


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


class DeclaredFieldsExtractor(BaseExtractor[Model]):
    """Build one document from the fields a model declares when registered."""

    def __init__(self, fields: tuple[str, ...], title_field: str | None) -> None:
        """Read ``fields`` as the text, ``title_field`` (if any) as the title."""
        self.fields = fields
        self.title_field = title_field

    def extract(self, instance: Model) -> NormalizedDocument | None:
        """Build the document of ``instance``, or nothing when its fields are blank."""
        field_texts = [_field_text(instance, name) for name in self.fields]
        text = _document_text(field_texts)
        if not text:
            return None
        return self.build_document(
            instance, text=text, title=self._title(instance, field_texts)
        )

    def _title(self, instance: Model, field_texts: list[str]) -> str:
        """Take the text of the title field, or the first of ``field_texts``.

        ``field_texts`` are the texts of the declared fields, in order: a title
        field among them is not read a second time.
        """
        if self.title_field is None:
            return field_texts[0]
        if self.title_field in self.fields:
            return field_texts[self.fields.index(self.title_field)]
        return _field_text(instance, self.title_field)
