"""Minimal Django settings for the test suite (derived from the prototype's)."""

SECRET_KEY = "tests-only-not-secret"
INSTALLED_APPS = ["tests.testapp"]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"
