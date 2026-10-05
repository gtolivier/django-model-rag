"""Autodiscovery of the ``model_rag`` module of each installed app at startup."""

import sys


def test_startup_imports_the_model_rag_module_of_each_app_having_one() -> None:
    # Django has already started when the test runs (pytest-django sets it
    # up), with contenttypes installed: an app without a model_rag module,
    # which must be skipped without an error. Nothing else imports these
    # modules, so finding them in sys.modules means startup imported them.
    assert "tests.testapp.model_rag" in sys.modules
    assert "tests.otherapp.model_rag" in sys.modules
