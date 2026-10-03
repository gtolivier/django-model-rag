"""Discover and normalize text content from any Django model."""

from django_model_rag.documents import NormalizedDocument
from django_model_rag.extractors import BaseExtractor
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import AlreadyRegistered, NotRegistered, rag

__all__ = [
    "AlreadyRegistered",
    "BaseExtractor",
    "NormalizedDocument",
    "NotRegistered",
    "SyncPipeline",
    "rag",
]
