"""The output protocol: where the pipeline hands over the documents it builds.

Also builds the output configured by the ``MODEL_RAG_OUTPUT`` setting.
"""

import inspect
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any, Protocol, TextIO

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from django_model_rag.documents import NormalizedDocument

_OUTPUT_SETTING = "MODEL_RAG_OUTPUT"
_BACKEND_KEY = "BACKEND"
_OPTIONS_KEY = "OPTIONS"
_REQUIRED_METHODS = ("replace", "prune")


class DocumentOutput(Protocol):
    """Receives the documents of the pipeline, grouped by source key."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Replace the stored documents of each source key with its group."""

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Delete the documents of ``model_label`` whose source key is not kept."""


class ConsoleOutput:
    """Writes the documents to a stream, standard output by default."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def _print(self, text: object) -> None:
        print(text, file=self._stream)

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Write each group: its source key, then its documents' titles and texts.

        An empty group means its source key's documents are removed: it is
        written as that removal.
        """
        for key, documents in groups.items():
            if not documents:
                self._print(f"{key} removed")
                continue
            self._print(key)
            for document in documents:
                self._print(document.title)
                self._print(document.text)

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Write the model label and the number of source keys kept."""
        self._print(f"{model_label} kept {len(kept_keys)}")


def configured_output() -> DocumentOutput:
    """Build the output named by the BACKEND of the output setting.

    Its OPTIONS, if any, are passed to that class as keyword arguments.

    Raises:
        ImproperlyConfigured: the output setting is missing or not a dict; its
            BACKEND is missing, not a string, cannot be imported, is not a
            class or lacks a callable ``replace`` or ``prune``; its OPTIONS
            are not a dict; or the BACKEND class's signature rejects its
            OPTIONS, which is checked before building it.
        Exception: whatever the BACKEND class raises while being built,
            unchanged. That includes its own TypeError for OPTIONS it rejects
            when its signature cannot be read, since they could not be checked
            beforehand.
    """
    output_class, options = _validated_output_configuration()
    return output_class(**options)


def check_output_configuration() -> None:
    """Fail unless the output setting names a usable output, building nothing."""
    _validated_output_configuration()


def _validated_output_configuration() -> tuple[type[DocumentOutput], dict[str, Any]]:
    """Return the configured output class and its OPTIONS, building nothing.

    Fail unless the output setting names a usable class that accepts them.
    """
    output_setting = _output_setting()
    output_class = _validated_output_class(output_setting)
    options = _options(output_setting)
    _require_accepted_options(output_class, options)
    return output_class, options


def _require_accepted_options(
    output_class: type[DocumentOutput], options: dict[str, Any]
) -> None:
    """Fail unless `output_class` can be built with `options`, building nothing."""
    try:
        signature = inspect.signature(output_class)
    except ValueError:
        # No readable signature (e.g. a C class): nothing to check against.
        return
    try:
        signature.bind(**options)
    except TypeError as error:
        message = f"The output {output_class.__name__} rejects its options: {error}"
        raise ImproperlyConfigured(message) from error


def _validated_output_class(output_setting: dict[str, Any]) -> type[DocumentOutput]:
    """Return the class named by the BACKEND of `output_setting`, if usable."""
    output_class = _backend_class(output_setting[_BACKEND_KEY])
    _require_callable_methods(output_class)
    return output_class


def _options(output_setting: dict[str, Any]) -> Any:
    """Return the OPTIONS of `output_setting`, empty if it has none."""
    return output_setting.get(_OPTIONS_KEY, {})


def _backend_class(backend: str) -> type[DocumentOutput]:
    """Import the output class at the dotted path `backend`.

    Fail unless that path can be imported and names a class.
    """
    try:
        output_class: type[DocumentOutput] = import_string(backend)
    except ImportError as error:
        message = f"The {_BACKEND_KEY} {backend} cannot be imported."
        raise ImproperlyConfigured(message) from error
    if not isinstance(output_class, type):
        message = f"The {_BACKEND_KEY} {backend} is not a class."
        raise ImproperlyConfigured(message)
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
    """Return the output setting, failing unless it is a dict with a BACKEND.

    Its BACKEND must be a string; its OPTIONS, if any, must be a dict too.
    """
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
    if not isinstance(output_setting[_BACKEND_KEY], str):
        message = (
            f"The {_BACKEND_KEY} of the {_OUTPUT_SETTING} setting must be a string."
        )
        raise ImproperlyConfigured(message)
    if not isinstance(_options(output_setting), dict):
        message = f"The {_OPTIONS_KEY} of the {_OUTPUT_SETTING} setting must be a dict."
        raise ImproperlyConfigured(message)
    return output_setting
