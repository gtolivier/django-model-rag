from collections.abc import Mapping, Sequence

from django_model_rag import DocumentOutput, NormalizedDocument


class RecordingOutput:
    """An output that records what the pipeline hands it."""

    def __init__(self) -> None:
        self.replaced: list[Mapping[str, Sequence[NormalizedDocument]]] = []
        self.pruned: list[tuple[str, set[str]]] = []

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        self.replaced.append(groups)

    def prune(self, model_label: str, kept_keys: set[str]) -> None:
        self.pruned.append((model_label, kept_keys))


def test_a_class_with_replace_and_prune_is_a_document_output() -> None:
    recorder = RecordingOutput()
    document = NormalizedDocument(
        text="A desk lamp",
        source_app_label="testapp",
        source_model="product",
        source_pk=1,
    )

    output: DocumentOutput = recorder
    output.replace({document.source_key: [document]})
    output.prune("testapp.product", {document.source_key})

    assert recorder.replaced == [{"testapp.product:1": [document]}]
    assert recorder.pruned == [("testapp.product", {"testapp.product:1"})]


class ReplaceOnlyOutput:
    """An output that can replace documents but not prune them."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        pass


def test_a_class_without_prune_is_not_a_document_output() -> None:
    # The type checker must reject the assignment: if the protocol ever
    # accepted a class without prune, warn_unused_ignores would flag this line.
    output: DocumentOutput = ReplaceOnlyOutput()  # type: ignore[assignment]

    assert not hasattr(output, "prune")
