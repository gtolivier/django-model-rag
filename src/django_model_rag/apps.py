"""Django app configuration: autodiscovery of the apps' ``model_rag`` modules."""

from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules


class ModelRagConfig(AppConfig):
    name = "django_model_rag"

    def ready(self) -> None:
        autodiscover_modules("model_rag")
