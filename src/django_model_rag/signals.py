"""Signal receivers that keep the output in step with saved instances."""

from typing import Any

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Model

from django_model_rag.output import configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import rag


def _committed_instance(model: type[Model], pk: Any) -> Model | None:
    """Return the instance of ``model`` with primary key ``pk`` as committed, if any."""
    try:
        return model._base_manager.get(pk=pk)
    except ObjectDoesNotExist:
        return None


def sync_saved_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a saved registered instance once its transaction commits."""
    if sender not in rag.registered_models():
        return

    # delete() clears the primary key of the instance: keep it for the commit.
    saved_pk = instance.pk

    def replace_group_as_committed() -> None:
        committed_instance = _committed_instance(sender, saved_pk)
        # Deleted since the save: the delete's own callback sends the empty group.
        if committed_instance is not None:
            SyncPipeline(configured_output()).run_instance(committed_instance)

    transaction.on_commit(replace_group_as_committed)


def sync_deleted_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a deleted registered instance with an empty one."""
    if sender not in rag.registered_models():
        return

    # delete() clears the primary key of the instance: keep it for the commit.
    deleted_pk = instance.pk

    def replace_group_with_an_empty_one() -> None:
        # A bare instance makes run_instance() send an empty group, as the row
        # is gone.
        SyncPipeline(configured_output()).run_instance(sender(pk=deleted_pk))

    transaction.on_commit(replace_group_with_an_empty_one)
