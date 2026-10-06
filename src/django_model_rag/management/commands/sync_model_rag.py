"""The `sync_model_rag` management command."""

import traceback
from typing import Any

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db.models import Model

from django_model_rag import SyncPipeline, configured_output, rag

_LABELS_ARGUMENT = "labels"
_VERBOSITY_OPTION = "verbosity"
_TRACEBACK_OPTION = "traceback"
_SILENT = 0


class Command(BaseCommand):
    help = "Synchronize the registered models into the configured output."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(_LABELS_ARGUMENT, nargs="*", metavar="app_label.model_name")

    def handle(self, *args: Any, **options: Any) -> None:
        pipeline = SyncPipeline(configured_output())
        failed_labels: list[str] = []
        models = _models_to_sync(options[_LABELS_ARGUMENT])
        if not models:
            self.stderr.write("Warning: no model is registered, nothing to sync.")
        for model in models:
            label = model._meta.label_lower
            try:
                pipeline.run([model])
            except Exception as error:  # one model's failure must not stop the others
                failed_labels.append(label)
                self._write_failure(label, error, options)
                continue
            if options[_VERBOSITY_OPTION] > _SILENT:
                self.stdout.write(f"{label}: synced")
        if failed_labels:
            message = f"Failed to sync: {', '.join(failed_labels)}."
            raise CommandError(message)

    def _write_failure(
        self, label: str, error: Exception, options: dict[str, Any]
    ) -> None:
        """Write the failure of model `label` on stderr, with its traceback if asked."""
        self.stderr.write(f"{label}: {type(error).__name__}: {error}")
        if options[_TRACEBACK_OPTION]:
            self.stderr.write("".join(traceback.format_exception(error)).rstrip())


def _models_to_sync(labels: list[str]) -> list[type[Model]]:
    """Return the models named by `labels`, or every registered model if none.

    A model named more than once is returned once, at its first position.
    """
    if not labels:
        return rag.registered_models()
    models = [_registered_model_named(label) for label in labels]
    return list(dict.fromkeys(models))


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
