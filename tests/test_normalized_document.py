import uuid
from collections.abc import Hashable
from typing import Any

import pytest

from django_model_rag import NormalizedDocument

MANDATORY_VALUES: dict[str, Any] = {
    "text": "A desk lamp",
    "source_app_label": "testapp",
    "source_model": "product",
    "source_pk": 1,
}


def make_document(**overrides: Any) -> NormalizedDocument:
    return NormalizedDocument(**{**MANDATORY_VALUES, **overrides})


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


def test_document_source_pk_cannot_be_none() -> None:
    # A document comes from a saved instance, which always has a primary key.
    with pytest.raises(ValueError, match="source_pk"):
        NormalizedDocument(
            text="A desk lamp",
            source_app_label="testapp",
            source_model="product",
            source_pk=None,
        )


def test_document_title_and_url_default_to_empty() -> None:
    document = make_document()

    assert document.title == ""
    assert document.url == ""


def test_document_language_is_unknown_by_default() -> None:
    document = make_document()

    assert document.language is None


def test_document_metadata_defaults_to_its_own_empty_dict() -> None:
    first = make_document()
    second = make_document(text="An office chair", source_pk=2)

    assert first.metadata == {}
    assert second.metadata == {}
    assert first.metadata is not second.metadata


def test_document_exposes_its_title_url_language_and_metadata() -> None:
    document = make_document(
        title="Desk lamp",
        url="/products/1/",
        language="en",
        metadata={"category": "Lighting"},
    )

    assert document.title == "Desk lamp"
    assert document.url == "/products/1/"
    assert document.language == "en"
    assert document.metadata == {"category": "Lighting"}


def test_document_keeps_its_own_copy_of_the_metadata_it_is_given() -> None:
    metadata = {"category": "Lighting"}
    document = make_document(metadata=metadata)

    metadata["category"] = "Furniture"

    assert document.metadata == {"category": "Lighting"}


def test_document_metadata_cannot_be_modified_through_the_document() -> None:
    document = make_document(metadata={"category": "Lighting"})

    # The assignment deliberately writes to a read-only Mapping: the type
    # checker rightly rejects it, and the test checks the runtime rejects it too.
    with pytest.raises(TypeError):
        document.metadata["category"] = "Furniture"  # type: ignore[index]

    assert document.metadata == {"category": "Lighting"}


def test_document_stores_the_permissions_it_is_given_as_a_frozenset() -> None:
    # Built without make_document so the type checker sees the set argument.
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        permissions={"testapp.view_product", "testapp.change_product"},
    )

    assert isinstance(document.permissions, frozenset)
    assert document.permissions == frozenset(
        {"testapp.view_product", "testapp.change_product"}
    )


def test_document_permissions_support_set_operations() -> None:
    # Typed as an abstract set of strings, so callers can compare and combine
    # the permissions with set operators without converting them first.
    document = make_document(permissions={"testapp.view_product"})
    view_or_change = {"testapp.view_product", "testapp.change_product"}
    view_or_delete = {"testapp.view_product", "testapp.delete_product"}

    assert document.permissions <= view_or_change
    assert document.permissions & view_or_delete == {"testapp.view_product"}


def test_document_refuses_a_bare_string_as_permissions() -> None:
    # A string is a collection of strings too: storing it would silently split
    # "testapp.view_product" into its single characters.
    with pytest.raises(TypeError, match="permissions"):
        make_document(permissions="testapp.view_product")


def test_document_refuses_permissions_holding_an_item_that_is_not_a_string() -> None:
    # Each bad item follows a valid one, so checking only the first item, or
    # only one kind of wrong item, is not enough.
    with pytest.raises(TypeError, match="permissions"):
        make_document(permissions=["testapp.view_product", 1])
    with pytest.raises(TypeError, match="permissions"):
        make_document(permissions=["testapp.view_product", None])


def test_document_keeps_the_permissions_it_is_given_as_a_one_shot_iterator() -> None:
    # A generator can be read only once: checking its items must not use it up
    # before the document stores them.
    permissions = (
        codename for codename in ["testapp.view_product", "testapp.change_product"]
    )
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        # A generator is outside the typed contract (a set of strings), but a
        # caller without a type checker can pass one and must not silently
        # lose its permissions.
        permissions=permissions,  # type: ignore[arg-type]
    )

    assert document.permissions == frozenset(
        {"testapp.view_product", "testapp.change_product"}
    )


def test_document_source_key_identifies_its_source() -> None:
    document = make_document(
        source_app_label="testapp", source_model="product", source_pk=1
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
        "text='A desk lamp with an adjustable arm, a weighted base and a wa'…>"
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


def test_document_repr_cuts_a_long_title_to_its_first_60_characters() -> None:
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
        title=(
            "Desk lamp with an adjustable arm, "
            "a weighted base and a warm white LED bulb"
        ),
    )

    assert repr(document) == (
        "<NormalizedDocument testapp.product:1 "
        "title='Desk lamp with an adjustable arm, a weighted base and a warm'… "
        "text='A desk lamp'>"
    )


DOCUMENT_VALUES: dict[str, Any] = {
    "text": "A desk lamp",
    "source_app_label": "testapp",
    "source_model": "product",
    "source_pk": 1,
    "title": "Lamp",
    "url": "/products/1/",
    "language": "en",
    "metadata": {"category": "Lighting"},
}


@pytest.mark.parametrize(
    ("field_name", "other_value"),
    [
        ("text", "An office chair"),
        ("source_app_label", "otherapp"),
        ("source_model", "category"),
        ("source_pk", 2),
        ("title", "Chair"),
        ("url", "/products/2/"),
        ("language", "fr"),
        ("metadata", {"category": "Furniture"}),
    ],
)
def test_documents_compare_by_value(field_name: str, other_value: object) -> None:
    document = NormalizedDocument(**DOCUMENT_VALUES)
    same = NormalizedDocument(**DOCUMENT_VALUES)
    different = NormalizedDocument(**{**DOCUMENT_VALUES, field_name: other_value})

    assert document == same
    assert document != different


@pytest.mark.parametrize("field_name", list(DOCUMENT_VALUES))
def test_document_attributes_cannot_be_reassigned(field_name: str) -> None:
    document = NormalizedDocument(**DOCUMENT_VALUES)

    # setattr rather than a plain assignment: the test covers every field, and
    # it type-checks the same before and after the class becomes immutable.
    with pytest.raises(AttributeError):
        setattr(document, field_name, "Other")


def test_document_is_not_hashable() -> None:
    # Documents compare by value but are not meant for sets or dict keys.
    document = NormalizedDocument(**DOCUMENT_VALUES)

    assert not isinstance(document, Hashable)
    with pytest.raises(TypeError):
        hash(document)
