import pytest
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command


def test_the_command_without_an_output_setting_names_the_missing_setting() -> None:
    # tests/settings.py defines no MODEL_RAG_OUTPUT.
    with pytest.raises(ImproperlyConfigured, match="MODEL_RAG_OUTPUT"):
        call_command("sync_model_rag")
