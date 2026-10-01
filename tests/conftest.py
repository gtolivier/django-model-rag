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


@pytest.fixture
def register() -> Iterator[Callable[..., None]]:
    """Register models with ``rag``; unregister them when the test ends."""
    registered: list[type[Model]] = []

    def register_model(model: type[Model], *, fields: list[str]) -> None:
        rag.register(model, fields=fields)
        registered.append(model)

    yield register_model
    for model in registered:
        rag.unregister(model)
