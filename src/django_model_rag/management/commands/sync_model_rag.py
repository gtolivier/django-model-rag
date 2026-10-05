"""The `sync_model_rag` management command."""

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def handle(self, *args: Any, **options: Any) -> None:
        if not hasattr(settings, "MODEL_RAG_OUTPUT"):
            message = "The MODEL_RAG_OUTPUT setting is required."
            raise ImproperlyConfigured(message)
