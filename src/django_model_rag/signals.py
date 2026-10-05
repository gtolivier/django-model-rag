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
    return [
        candidate for candidate in candidates if candidate in rag.registered_models()
    ]


def _registered_model(sender: type[Model]) -> type[Model] | None:
    """Return the nearest registered model whose group ``sender`` feeds, if any."""
    registered_models = _registered_models(sender)
    return registered_models[0] if registered_models else None


def _replace_group(instance: Model) -> None:
    """Send the group of ``instance`` to a newly built configured output."""
    SyncPipeline(configured_output()).run_instance(instance)


def sync_saved_instance(
    sender: type[Model], instance: Model, raw: bool = False, **kwargs: Any
) -> None:
    """Replace the group of a saved registered instance once its transaction commits."""
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    if not registered_models:
        return

    # Fail at the save, not at the commit, if the output is misconfigured.
    check_output_configuration()

    # delete() clears the primary key of the instance: keep it for the commit.
    saved_pk = instance.pk

    for registered_model in registered_models:
        transaction.on_commit(_group_replacer(registered_model, saved_pk))


def _group_replacer(registered_model: type[Model], saved_pk: Any) -> Callable[[], None]:
    """Return a commit callback replacing the group of a saved instance."""

    def replace_group_as_committed() -> None:
        committed_instance = _committed_instance(registered_model, saved_pk)
        # Deleted since the save: the delete's own callback sends the empty group.
        if committed_instance is None:
            return

        try:
            _replace_group(committed_instance)
        except Exception:
            # An error escaping a commit callback would break the commit.
            logger.exception(
                "Syncing %s failed",
                build_source_key(registered_model._meta.label_lower, saved_pk),
            )

    return replace_group_as_committed


def sync_deleted_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a deleted registered instance with an empty one."""
    if not _signals_enabled():
        return

    registered_model = _registered_model(sender)
    if registered_model is None:
        return

    # Fail at the delete, not at the commit, if the output is misconfigured.
    check_output_configuration()

    # delete() clears the primary key of the instance: keep it for the commit.
    deleted_pk = instance.pk

    def replace_group_with_an_empty_one() -> None:
        # A bare instance makes run_instance() send an empty group, as the row
        # is gone.
        _replace_group(registered_model(pk=deleted_pk))

    transaction.on_commit(replace_group_with_an_empty_one)
