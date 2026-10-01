import pytest

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


def test_document_source_is_mandatory() -> None:
    # Each call deliberately omits one source field: the type checker rightly
    # rejects it, and the test checks the runtime rejects it too.
    with pytest.raises(TypeError):
        NormalizedDocument(  # type: ignore[call-arg]
            text="A desk lamp",
            source_model="product",
            source_pk=1,
        )
    with pytest.raises(TypeError):
        NormalizedDocument(  # type: ignore[call-arg]
            text="A desk lamp",
            source_app_label="testapp",
            source_pk=1,
        )
    with pytest.raises(TypeError):
        NormalizedDocument(  # type: ignore[call-arg]
            text="A desk lamp",
            source_app_label="testapp",
            source_model="product",
        )


def test_document_arguments_are_keyword_only() -> None:
    # The call deliberately passes the arguments positionally: the type checker
    # rightly rejects it, and the test checks the runtime rejects it too.
    with pytest.raises(TypeError):
        NormalizedDocument("A desk lamp", "testapp", "product", 1)  # type: ignore[call-arg]


def test_document_title_and_url_default_to_empty() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )

    assert document.title == ""
    assert document.url == ""


def test_document_language_is_unknown_by_default() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )

    assert document.language is None
