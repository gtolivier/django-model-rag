from collections.abc import Callable, Iterator

import pytest
from django.db import connection
from django.db.models import Model

from django_model_rag import rag


@pytest.fixture
def unordered_selects_reversed(db: None) -> Iterator[None]:
    """Make SQLite return the rows of any query without ORDER BY in reverse.

    SQLite otherwise tends to return rows in primary key order anyway, which
    would hide a query that relies on that instead of ordering explicitly.
    """
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA reverse_unordered_selects = ON")
    yield
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA reverse_unordered_selects = OFF")


def registered_models() -> set[type[Model]]:
    """Return the models currently registered with ``rag``."""
    return {model for model, _fields in rag.declarations()}


@pytest.fixture
def restored_registry() -> Iterator[None]:
    """Unregister, when the test ends, every model it left registered with ``rag``.

    Only models still registered are unregistered, so a test may unregister
    a model itself without making the teardown fail.
    """
    registered_before = registered_models()
    yield
    for model in registered_models() - registered_before:
        rag.unregister(model)


@pytest.fixture
def register(restored_registry: None) -> Callable[..., None]:
    """Register models with ``rag``; unregister them when the test ends."""
    return rag.register
