import sqlite3
from collections.abc import Iterator
from typing import TypeVar

import pytest
from django.db import connection

from django_model_rag import rag
from tests.recording import PruneOnlyOutput, ReplaceOnlyOutput, TrackedRecordingOutput

T = TypeVar("T")


def _only_built_during_the_test(built: list[T]) -> Iterator[list[T]]:
    """Yield a class's `built` list emptied, and empty it again after the test."""
    built.clear()
    yield built
    built.clear()


@pytest.fixture
def built_outputs() -> Iterator[list[TrackedRecordingOutput]]:
    """The TrackedRecordingOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(TrackedRecordingOutput.built)


@pytest.fixture
def built_prune_only_outputs() -> Iterator[list[PruneOnlyOutput]]:
    """The PruneOnlyOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(PruneOnlyOutput.built)


@pytest.fixture
def built_replace_only_outputs() -> Iterator[list[ReplaceOnlyOutput]]:
    """The ReplaceOnlyOutput instances built during the test, and only those."""
    yield from _only_built_during_the_test(ReplaceOnlyOutput.built)


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
def sqlite_variable_limit_lowered(db: None) -> Iterator[int]:
    """Lower the number of variables SQLite accepts in one query, and yield it.

    The local SQLite build accepts tens of thousands of them: lowered, a query
    that puts one variable per row in its SQL fails with a few rows only. The
    previous limit is restored after the test.
    """
    lowered_limit = 1500
    connection.ensure_connection()
    sqlite_connection = connection.connection
    previous_limit = sqlite_connection.setlimit(
        sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, lowered_limit
    )
    yield lowered_limit
    sqlite_connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)


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
