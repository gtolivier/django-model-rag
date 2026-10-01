from django_model_rag import SyncPipeline


def test_pipeline_without_registered_model_produces_no_document() -> None:
    assert SyncPipeline().run() == []
