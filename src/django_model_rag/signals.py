"""Signal receivers that keep the output in step with saved instances."""

from typing import Any

from django.db import transaction
from django.db.models import Model

from django_model_rag.output import configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import rag


def sync_saved_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Replace the group of a saved registered instance once its transaction commits."""
    if sender not in rag.registered_models():
        return
    transaction.on_commit(
        lambda: SyncPipeline(configured_output()).run_instance(instance)
    )
