import uuid

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


def test_document_metadata_defaults_to_its_own_empty_dict() -> None:
    first = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )
    second = NormalizedDocument(
        text="An office chair",
        source_app_label="testapp",
        source_model="product",
        source_pk=2,
    )

    assert first.metadata == {}
    assert second.metadata == {}
    assert first.metadata is not second.metadata


def test_document_exposes_its_title_url_language_and_metadata() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        title="Desk lamp",
        url="/products/1/",
        language="en",
        metadata={"category": "Lighting"},
    )

    assert document.title == "Desk lamp"
    assert document.url == "/products/1/"
    assert document.language == "en"
    assert document.metadata == {"category": "Lighting"}


def test_document_source_key_identifies_its_source() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )

    assert document.source_key == "testapp.product:1"


def test_document_source_key_uses_the_string_form_of_any_primary_key() -> None:
    page_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    uuid_document = NormalizedDocument(
        text="Welcome",
        source_app_label="testapp",
        source_model="page",
        source_pk=page_uuid,
    )
    string_document = NormalizedDocument(
        text="Welcome",
        source_app_label="testapp",
        source_model="page",
        source_pk="intro",
    )

    assert uuid_document.source_key == (
        "testapp.page:12345678-1234-5678-1234-567812345678"
    )
    assert string_document.source_key == "testapp.page:intro"


def test_document_repr_shows_its_source_key_title_and_text() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        title="Lamp",
    )

    assert repr(document) == (
        "<NormalizedDocument testapp.product:1 title='Lamp' text='A desk lamp'>"
    )


def test_document_repr_cuts_a_long_text_to_its_first_60_characters() -> None:
    document = NormalizedDocument(
        text=(
            "A desk lamp with an adjustable arm, "
            "a weighted base and a warm white LED bulb."
        ),
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        title="Lamp",
    )

    assert repr(document) == (
        "<NormalizedDocument testapp.product:1 title='Lamp' "
        "text='A desk lamp with an adjustable arm, a weighted base and a wa…'>"
    )


def test_document_repr_shows_a_text_of_exactly_60_characters_whole() -> None:
    text = "A desk lamp with an adjustable arm and a weighted metal base"
    assert len(text) == 60
    document = NormalizedDocument(
        text=text,
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        title="Lamp",
    )

    assert repr(document) == (
        "<NormalizedDocument testapp.product:1 title='Lamp' "
        "text='A desk lamp with an adjustable arm and a weighted metal base'>"
    )
