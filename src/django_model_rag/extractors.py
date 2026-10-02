"""The base class of custom extractors."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from django.db.models import Model

from django_model_rag.documents import NormalizedDocument

M = TypeVar("M", bound=Model)


def document_from_instance(  # noqa: PLR0913  # one keyword per document field
    instance: Model,
    *,
    text: str,
    title: str = "",
    url: str = "",
    language: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NormalizedDocument:
    """Build a document of ``text`` and its details, sourced from ``instance``."""
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
        return document_from_instance(
            instance,
            text=text,
            title=title,
            url=url,
            language=language,
            metadata=metadata,
        )
