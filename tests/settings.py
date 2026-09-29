"""Minimal Django settings for the test suite (derived from the prototype's)."""

SECRET_KEY = "tests-only-not-secret"
INSTALLED_APPS = ["tests.testapp"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
# Pinned: the default changed to BigAutoField in Django 6.0, so leaving it
# unset would make the test app's migration drift between supported versions.
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
