from django_model_rag import rag


def test_registered_models_is_empty_when_no_model_is_registered() -> None:
    assert list(rag.registered_models()) == []
