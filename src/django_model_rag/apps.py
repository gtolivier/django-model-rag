"""Django app configuration: autodiscovery of the apps' ``model_rag`` modules."""

from django.apps import AppConfig
from django.db.models.signals import m2m_changed, post_save, pre_save
from django.utils.module_loading import autodiscover_modules

DISCOVERED_MODULE = "model_rag"
"""The module each installed app registers its models from."""


class ModelRagConfig(AppConfig):
    name = "django_model_rag"

    def ready(self) -> None:
        from django_model_rag.signals import (
            check_output_before_save,
            remember_followers_before_save,
            sync_changed_relation,
            sync_saved_instance,
        )

        pre_save.connect(
            remember_followers_before_save,
            dispatch_uid="django_model_rag.remember_followers",
        )
        pre_save.connect(
            check_output_before_save, dispatch_uid="django_model_rag.check_output"
        )
        post_save.connect(sync_saved_instance, dispatch_uid="django_model_rag.sync")
        m2m_changed.connect(
            sync_changed_relation, dispatch_uid="django_model_rag.sync_relation"
        )
        autodiscover_modules(DISCOVERED_MODULE)
