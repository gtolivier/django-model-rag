"""The `sync_model_rag` management command."""

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand
from django.utils.module_loading import import_string

from django_model_rag import DocumentOutput, SyncPipeline

_OUTPUT_SETTING = "MODEL_RAG_OUTPUT"
_BACKEND_KEY = "BACKEND"
_OPTIONS_KEY = "OPTIONS"


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def handle(self, *args: Any, **options: Any) -> None:
        SyncPipeline(_configured_output()).run()


def _configured_output() -> DocumentOutput:
    """Build the output named by the BACKEND of the output setting.

    Its OPTIONS, if any, are passed to that class as keyword arguments.
    """
    if not hasattr(settings, _OUTPUT_SETTING):
        message = f"The {_OUTPUT_SETTING} setting is required."
        raise ImproperlyConfigured(message)
    output_setting = getattr(settings, _OUTPUT_SETTING)
    if not isinstance(output_setting, dict):
        message = f"The {_OUTPUT_SETTING} setting must be a dict."
        raise ImproperlyConfigured(message)
    output_class: type[DocumentOutput] = import_string(output_setting[_BACKEND_KEY])
    return output_class(**output_setting.get(_OPTIONS_KEY, {}))
