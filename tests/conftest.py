from collections.abc import Callable, Iterator

import pytest
from django.db.models import Model

from django_model_rag import rag


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
