"""The `sync_model_rag` management command."""

from typing import Any

from django.apps import apps
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db.models import Model
from django.utils.module_loading import import_string

from django_model_rag import DocumentOutput, SyncPipeline, rag

_LABELS_ARGUMENT = "labels"
_OUTPUT_SETTING = "MODEL_RAG_OUTPUT"
_BACKEND_KEY = "BACKEND"
_OPTIONS_KEY = "OPTIONS"
_REQUIRED_METHODS = ("replace", "prune")


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(_LABELS_ARGUMENT, nargs="*", metavar="app_label.model_name")

    def handle(self, *args: Any, **options: Any) -> None:
        pipeline = SyncPipeline(_configured_output())
        failed_labels: list[str] = []
        for model in _models_to_sync(options[_LABELS_ARGUMENT]):
            label = model._meta.label_lower
            try:
                pipeline.run([model])
            except Exception:  # one model's failure must not stop the others
                failed_labels.append(label)
                continue
            self.stdout.write(f"{label}: synced")
        if failed_labels:
            message = f"Failed to sync: {', '.join(failed_labels)}."
            raise CommandError(message)


def _models_to_sync(labels: list[str]) -> list[type[Model]]:
    """Return the models named by `labels`, or every registered model if none."""
    if not labels:
        return rag.registered_models()
    return [_registered_model_named(label) for label in labels]


def _registered_model_named(label: str) -> type[Model]:
    """Return the registered model named by `label`, else fail with CommandError."""
    model = _model_named(label)
    if model not in rag.registered_models():
        message = f"The model {model._meta.label_lower} is not registered."
        raise CommandError(message)
    return model


def _model_named(label: str) -> type[Model]:
    """Return the model named by `label`, failing with CommandError if none."""
    try:
        return apps.get_model(label)
    except (LookupError, ValueError) as error:
        message = f"The label {label} names no model of an installed app."
        raise CommandError(message) from error


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
