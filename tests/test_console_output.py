import pytest

from django_model_rag import NormalizedDocument
from django_model_rag.output import ConsoleOutput


def test_console_output_writes_each_group_key_then_its_documents_titles_and_texts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shipping = NormalizedDocument(
        text="We ship within two days.",
        source_app_label="testapp",
        source_model="page",
        source_pk=7,
        title="Shipping",
    )
    refunds = NormalizedDocument(
        text="Returns are free for thirty days.",
        source_app_label="testapp",
        source_model="page",
        source_pk=7,
        title="Refunds",
    )
    history = NormalizedDocument(
        text="Founded in a garage.",
        source_app_label="testapp",
        source_model="page",
        source_pk=42,
        title="History",
    )

    ConsoleOutput().replace(
        {
            "testapp.page:7": [shipping, refunds],
            "testapp.page:42": [history],
        }
    )

    # No exact layout is frozen: each fragment must appear after the previous
    # one, so a group's documents follow its key, in order, before the next
    # group's key.
    written = capsys.readouterr().out
    expected_in_order = [
        "testapp.page:7",
        "Shipping",
        "We ship within two days.",
        "Refunds",
        "Returns are free for thirty days.",
        "testapp.page:42",
        "History",
        "Founded in a garage.",
    ]
    position = 0
    for fragment in expected_in_order:
        found = written.find(fragment, position)
        assert found != -1, f"{fragment!r} missing after position {position}"
        position = found + len(fragment)
