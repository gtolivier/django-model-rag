"""Django app configuration: autodiscovery of the apps' ``model_rag`` modules."""

from django.apps import AppConfig
from django.db.models.signals import post_save, pre_save
from django.utils.module_loading import autodiscover_modules

DISCOVERED_MODULE = "model_rag"
"""The module each installed app registers its models from."""


class ModelRagConfig(AppConfig):
    name = "django_model_rag"

    def ready(self) -> None:
        from django_model_rag.signals import (
            check_output_before_save,
            sync_saved_instance,
        )

        pre_save.connect(
            check_output_before_save, dispatch_uid="django_model_rag.check_output"
        )
        post_save.connect(sync_saved_instance, dispatch_uid="django_model_rag.sync")
        autodiscover_modules(DISCOVERED_MODULE)
