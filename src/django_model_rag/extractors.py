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
