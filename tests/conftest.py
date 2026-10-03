from collections.abc import Iterator

import pytest
from django.db import connection

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


@pytest.fixture(autouse=True)
def restored_registry() -> Iterator[None]:
    """Unregister, when any test ends, every model it left registered with ``rag``.

    Autouse, so that no test leaves a registration behind it and every test
    starts with the registry the previous one found.

    ``registered_models`` lists the models of every kind of registration
    (declared fields, custom extractors...), so the teardown undoes them all
    through the public API. Only models still registered are unregistered,
    so a test may unregister a model itself without making the teardown fail.
    """
    registered_before = set(rag.registered_models())
    yield
    for model in rag.registered_models():
        if model not in registered_before:
            rag.unregister(model)
