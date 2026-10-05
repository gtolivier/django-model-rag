"""Django app configuration: autodiscovery of the apps' ``model_rag`` modules."""

from django.apps import AppConfig
from django.db.models.signals import post_delete, post_save
from django.utils.module_loading import autodiscover_modules

DISCOVERED_MODULE = "model_rag"
"""The module each installed app registers its models from."""


class ModelRagConfig(AppConfig):
    name = "django_model_rag"

    def ready(self) -> None:
        from django_model_rag.signals import (
            sync_deleted_instance,
            sync_saved_instance,
        )

        post_save.connect(sync_saved_instance, dispatch_uid="django_model_rag.sync")
        post_delete.connect(
            sync_deleted_instance, dispatch_uid="django_model_rag.sync_delete"
        )
        autodiscover_modules(DISCOVERED_MODULE)
