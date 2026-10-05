"""Signal receivers that keep the output in step with saved instances."""

from typing import Any

from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import Model

from django_model_rag.output import configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import rag


def _committed_instance(model: type[Model], pk: Any) -> Model:
    """Return the instance of ``model`` with primary key ``pk`` as committed."""
    try:
        return model._base_manager.get(pk=pk)
    except ObjectDoesNotExist:
        # Deleted since the save: a bare instance makes run_instance() send
        # an empty group, as the row is gone.
        return model(pk=pk)


def sync_saved_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a saved registered instance once its transaction commits."""
    if sender not in rag.registered_models():
        return

    # delete() clears the primary key of the instance: keep it for the commit.
    saved_pk = instance.pk

    def replace_group_as_committed() -> None:
        committed_instance = _committed_instance(sender, saved_pk)
        SyncPipeline(configured_output()).run_instance(committed_instance)

    transaction.on_commit(replace_group_as_committed)
