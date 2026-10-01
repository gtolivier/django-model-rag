from django_model_rag import NormalizedDocument


def test_document_exposes_its_text_and_source() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )

    assert document.text == "A desk lamp"
    assert document.source_app_label == "testapp"
    assert document.source_model == "product"
    assert document.source_pk == 1
