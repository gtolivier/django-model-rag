"""The `sync_model_rag` management command."""

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand
from django.utils.module_loading import import_string

from django_model_rag import DocumentOutput, SyncPipeline, rag

_OUTPUT_SETTING = "MODEL_RAG_OUTPUT"
_BACKEND_KEY = "BACKEND"
_OPTIONS_KEY = "OPTIONS"
_REQUIRED_METHODS = ("replace", "prune")


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def handle(self, *args: Any, **options: Any) -> None:
        pipeline = SyncPipeline(_configured_output())
        for model in rag.registered_models():
            pipeline.run([model])
            self.stdout.write(f"{model._meta.label_lower}: synced")


def _configured_output() -> DocumentOutput:
    """Build the output named by the BACKEND of the output setting.

    Its OPTIONS, if any, are passed to that class as keyword arguments.
    """
    output_setting = _output_setting()
    output_class = _backend_class(output_setting[_BACKEND_KEY])
    _require_callable_methods(output_class)
    return output_class(**output_setting.get(_OPTIONS_KEY, {}))


def _backend_class(backend: str) -> type[DocumentOutput]:
    """Import the output class at the dotted path `backend`."""
    try:
        output_class: type[DocumentOutput] = import_string(backend)
    except ImportError as error:
        message = f"The {_BACKEND_KEY} {backend} cannot be imported."
        raise ImproperlyConfigured(message) from error
    return output_class


def _require_callable_methods(output_class: type[DocumentOutput]) -> None:
    """Fail if `output_class` lacks a callable for any of `_REQUIRED_METHODS`."""
    for method_name in _REQUIRED_METHODS:
        if not callable(getattr(output_class, method_name, None)):
            message = (
                f"The output {output_class.__name__} needs a callable {method_name}."
            )
            raise ImproperlyConfigured(message)


def _output_setting() -> dict[str, Any]:
    """Return the output setting, failing if it is not a dict with a BACKEND."""
    if not hasattr(settings, _OUTPUT_SETTING):
        message = f"The {_OUTPUT_SETTING} setting is required."
        raise ImproperlyConfigured(message)
    output_setting = getattr(settings, _OUTPUT_SETTING)
    if not isinstance(output_setting, dict):
        message = f"The {_OUTPUT_SETTING} setting must be a dict."
        raise ImproperlyConfigured(message)
    if _BACKEND_KEY not in output_setting:
        message = f"The {_OUTPUT_SETTING} setting requires a {_BACKEND_KEY} key."
        raise ImproperlyConfigured(message)
    return output_setting
