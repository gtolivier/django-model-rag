"""The `sync_model_rag` management command."""

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand

_OUTPUT_SETTING = "MODEL_RAG_OUTPUT"


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def handle(self, *args: Any, **options: Any) -> None:
        if not hasattr(settings, _OUTPUT_SETTING):
            message = f"The {_OUTPUT_SETTING} setting is required."
            raise ImproperlyConfigured(message)
