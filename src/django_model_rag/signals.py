"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable
from typing import Any

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Model

from django_model_rag.documents import build_source_key
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"

logger = logging.getLogger("django_model_rag")


def _signals_enabled() -> bool:
    """Return whether the signal receivers sync anything, as the settings say."""
    return bool(getattr(settings, _SIGNALS_SETTING, True))


def _committed_instance(model: type[Model], pk: Any) -> Model | None:
    """Return the instance of ``model`` with primary key ``pk`` as committed, if any."""
    try:
        return model._base_manager.get(pk=pk)
    except ObjectDoesNotExist:
        return None


def _registered_models(sender: type[Model]) -> list[type[Model]]:
    """Return the registered models whose groups ``sender``'s instances feed."""
    # A proxy sends signals under its own sender: its group is the concrete model's.
    concrete_model = sender._meta.concrete_model
    if concrete_model is None:
        return []
    # A multi-table child feeds the group of its registered parents too.
    candidates: tuple[type[Model], ...] = (
        concrete_model,
        *concrete_model._meta.get_parent_list(),
    )
    registered = set(rag.registered_models())
    return [candidate for candidate in candidates if candidate in registered]


def _schedule_commit_callbacks(
    registered_models: list[type[Model]],
    pk: Any,
    build_callback: Callable[[type[Model], Any], Callable[[], None]],
) -> None:
    """Run, at the commit, the callback ``build_callback`` returns per registered model.

    Each callback gets ``pk`` as it is now: delete() clears the primary key of
    the instance before the commit. Nothing runs while the signals are off.
    """
    if not registered_models or not _signals_enabled():
        return

    # Fail at the save or the delete, not at the commit, if the output is
    # misconfigured.
    check_output_configuration()

    for registered_model in registered_models:
        transaction.on_commit(build_callback(registered_model, pk))


def _source_key(registered_model: type[Model], pk: Any) -> str:
    """Return the source key of the group of ``registered_model`` for ``pk``."""
    return build_source_key(registered_model._meta.label_lower, pk)


def _send_group(send: Callable[[], None], source_key: str) -> None:
    """Run ``send``, logging a failure with the ``source_key`` of the group."""
    try:
        send()
    except Exception:
        # An error escaping a commit callback would break the commit.
        logger.exception("Syncing %s failed", source_key)


def _replace_group(instance: Model, registered_model: type[Model], pk: Any) -> None:
    """Send the group of ``instance`` to a newly built configured output."""
    _send_group(
        lambda: SyncPipeline(configured_output()).run_instance(instance),
        _source_key(registered_model, pk),
    )


def sync_saved_instance(
    sender: type[Model], instance: Model, raw: bool = False, **kwargs: Any
) -> None:
    """Replace the group of a saved registered instance once its transaction commits."""
    if raw:
        return

    _schedule_commit_callbacks(_registered_models(sender), instance.pk, _group_replacer)


def _group_replacer(registered_model: type[Model], saved_pk: Any) -> Callable[[], None]:
    """Return a commit callback replacing the group of a saved instance."""

    def replace_group_as_committed() -> None:
        def reload_and_replace() -> None:
            committed_instance = _committed_instance(registered_model, saved_pk)
            # Deleted since the save: the delete's own callback sends the empty group.
            if committed_instance is None:
                return

            _replace_group(committed_instance, registered_model, saved_pk)

        _send_group(reload_and_replace, _source_key(registered_model, saved_pk))

    return replace_group_as_committed


def sync_deleted_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a deleted registered instance with an empty one."""
    # Django sends post_delete for each multi-table parent too, under its own
    # sender: the nearest registered model is the only group to empty here.
    nearest_registered_model_only = _registered_models(sender)[:1]
    _schedule_commit_callbacks(
        nearest_registered_model_only, instance.pk, _group_emptier
    )


def _group_emptier(
    registered_model: type[Model], deleted_pk: Any
) -> Callable[[], None]:
    """Return a commit callback emptying the group of a deleted instance."""

    def replace_group_with_an_empty_one() -> None:
        # The row is gone: the empty group goes straight to the output, with no
        # extractor involved.
        source_key = _source_key(registered_model, deleted_pk)
        _send_group(lambda: configured_output().replace({source_key: []}), source_key)

    return replace_group_with_an_empty_one
