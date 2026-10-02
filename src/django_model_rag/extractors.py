"""The base class of custom extractors."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Generic, TypeVar

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

    def build_document(self, instance: M, *, text: str) -> NormalizedDocument:
        """Build a document with ``text``, its source taken from ``instance``."""
        app_label, model_name = instance._meta.label_lower.split(".")
        return NormalizedDocument(
            text=text,
            source_app_label=app_label,
            source_model=model_name,
            source_pk=instance.pk,
        )
