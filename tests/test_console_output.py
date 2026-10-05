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


def test_console_output_states_the_removal_of_each_empty_group_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = NormalizedDocument(
        text="Founded in a garage.",
        source_app_label="testapp",
        source_model="page",
        source_pk=42,
        title="History",
    )

    ConsoleOutput().replace(
        {
            "testapp.page:7": [],
            "testapp.page:42": [history],
            "testapp.product:13": [],
        }
    )

    # An empty group is the pipeline's way of saying the key's documents must
    # be removed: the line naming each such key says so, the line naming a key
    # with documents does not.
    lines = capsys.readouterr().out.splitlines()
    for removed_key in ("testapp.page:7", "testapp.product:13"):
        assert any(removed_key in line and "removed" in line for line in lines), (
            f"no line states the removal of {removed_key!r}"
        )
    kept_lines = [line for line in lines if "testapp.page:42" in line]
    assert kept_lines, "the key with documents is not written"
    assert not any("removed" in line for line in kept_lines)
