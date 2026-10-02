"""The base class of custom extractors."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument

M = TypeVar("M", bound=Model)


def document_from_instance(
    instance: Model, *, text: str, title: str = ""
) -> NormalizedDocument:
    """Build a document of ``text`` and ``title``, sourced from ``instance``."""
    return NormalizedDocument(
        text=text,
        source_app_label=instance._meta.app_label,  # Django's public meta API
        source_model=instance._meta.model_name or "",  # Django's public meta API
        source_pk=instance.pk,
        title=title,
    )


class BaseExtractor(ABC, Generic[M]):
    """Build the document(s) of an instance of a model."""

    @abstractmethod
    def extract(
        self, instance: M
    ) -> NormalizedDocument | Iterable[NormalizedDocument] | None:
        """Build the document(s) of ``instance``, or nothing to skip it."""

    def build_document(self, instance: M, *, text: str) -> NormalizedDocument:
        """Build a document with ``text``, its source taken from ``instance``."""
        return document_from_instance(instance, text=text)
