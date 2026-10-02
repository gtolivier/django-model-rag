"""The base class of custom extractors."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument

M = TypeVar("M", bound=Model)


class BaseExtractor(ABC, Generic[M]):
    """Build the document of an instance of a model."""

    @abstractmethod
    def extract(
        self, instance: M
    ) -> NormalizedDocument | list[NormalizedDocument] | None:
        """Build the document(s) of ``instance``, or nothing to skip it."""
