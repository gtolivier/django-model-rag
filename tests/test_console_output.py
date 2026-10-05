import io
import re

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


def test_console_output_prune_writes_each_model_label_with_its_kept_key_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    kept_pages = {"testapp.page:7", "testapp.page:42", "testapp.page:99"}

    output = ConsoleOutput()
    output.prune("testapp.page", kept_pages)
    output.prune("testapp.product", set())

    # No exact wording is frozen: each label must share a line with its own
    # count and no other number, so a constant count cannot pass; the kept
    # keys themselves are not listed.
    written = capsys.readouterr().out
    lines = written.splitlines()
    for label, count in (("testapp.page", "3"), ("testapp.product", "0")):
        assert any(
            label in line and re.findall(r"\d+", line) == [count] for line in lines
        ), f"no line writes {label!r} with its count {count}"
    for key in kept_pages:
        assert key not in written, f"kept key {key!r} is written"


def test_console_output_writes_replace_and_prune_to_the_given_stream(
    capsys: pytest.CaptureFixture[str],
) -> None:
    history = NormalizedDocument(
        text="Founded in a garage.",
        source_app_label="testapp",
        source_model="page",
        source_pk=42,
        title="History",
    )
    stream = io.StringIO()

    output = ConsoleOutput(stream=stream)
    output.replace({"testapp.page:42": [history]})
    output.prune("testapp.product", {"testapp.product:13"})

    written = stream.getvalue()
    assert "testapp.page:42" in written
    assert "testapp.product" in written
    assert capsys.readouterr().out == ""
