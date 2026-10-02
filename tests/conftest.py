import copy
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
    """Put ``rag`` back, when any test ends, in the state the test found it in.

    Autouse, so that no test leaves a registration behind it and every test
    starts with the registry the previous one found.

    The whole state of ``rag`` is copied and restored, rather than each
    registration undone through the public API: that way the teardown covers
    every kind of registration (declared fields, custom extractors...)
    without knowing how each is stored, and a test may unregister a model
    itself without making the teardown fail.
    """
    state = copy.deepcopy(vars(rag))
    yield
    vars(rag).clear()
    vars(rag).update(state)
