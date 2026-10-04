from collections.abc import Mapping, Sequence

import pytest

from django_model_rag import (
    BaseExtractor,
    DocumentOutput,
    NormalizedDocument,
    SyncPipeline,
    rag,
)
from tests.recording import RecordingOutput
from tests.testapp.models import AccordionItem, Category, Page


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


class FlatReplaceOutput:
    """An output whose replace takes the documents one after the other."""

    def replace(self, documents: Sequence[NormalizedDocument]) -> None:
        pass

    def prune(self, model_label: str, kept_keys: set[str]) -> None:
        pass


def test_a_class_whose_replace_takes_a_flat_sequence_is_not_a_document_output() -> None:
    # The type checker must reject the assignment: replace receives the
    # documents grouped by source key, not one after the other. If the
    # protocol ever accepted a flat sequence, warn_unused_ignores would flag
    # this line.
    output: DocumentOutput = FlatReplaceOutput()  # type: ignore[assignment]

    assert hasattr(output, "prune")


@pytest.mark.django_db
def test_pipeline_hands_each_instance_documents_to_its_output_by_source_key() -> None:
    # Two documents for one page: a group per document, rather than per
    # source key, would lose one of them.
    faq = Page.objects.create(title="FAQ", slug="faq")
    AccordionItem.objects.create(
        page=faq, title="Shipping", body="We ship within two days."
    )
    AccordionItem.objects.create(
        page=faq, title="Returns", body="Returns are free for thirty days."
    )
    about = Page.objects.create(title="About", slug="about")
    AccordionItem.objects.create(
        page=about, title="History", body="Founded in a garage."
    )

    @rag.register_extractor(Page)
    class PageExtractor(BaseExtractor[Page]):
        def extract(self, instance: Page) -> list[NormalizedDocument]:
            return [
                self.build_document(instance, text=item.body)
                for item in instance.accordion_items.order_by("pk")
            ]

    run_output = RecordingOutput()
    SyncPipeline(run_output).run()
    instance_output = RecordingOutput()
    SyncPipeline(output=instance_output).run_instance(about)

    # run() may split its groups across several calls: merge them.
    run_groups = {
        source_key: group
        for groups in run_output.replaced
        for source_key, group in groups.items()
    }
    assert run_groups == {
        f"testapp.page:{faq.pk}": [
            NormalizedDocument(
                text="We ship within two days.",
                source_app_label="testapp",
                source_model="page",
                source_pk=faq.pk,
            ),
            NormalizedDocument(
                text="Returns are free for thirty days.",
                source_app_label="testapp",
                source_model="page",
                source_pk=faq.pk,
            ),
        ],
        f"testapp.page:{about.pk}": [
            NormalizedDocument(
                text="Founded in a garage.",
                source_app_label="testapp",
                source_model="page",
                source_pk=about.pk,
            ),
        ],
    }
    assert instance_output.replaced == [
        {
            f"testapp.page:{about.pk}": [
                NormalizedDocument(
                    text="Founded in a garage.",
                    source_app_label="testapp",
                    source_model="page",
                    source_pk=about.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_pipeline_hands_no_group_for_an_instance_without_documents() -> None:
    # An empty group would tell the output to replace that instance's
    # documents with nothing: deleting them is prune's job, not replace's.
    empty = Category.objects.create(name="")
    lighting = Category.objects.create(name="Lighting")
    rag.register(Category, fields=["name"])

    output = RecordingOutput()
    SyncPipeline(output).run()

    received_keys = {key for groups in output.replaced for key in groups}
    assert f"testapp.category:{lighting.pk}" in received_keys
    assert f"testapp.category:{empty.pk}" not in received_keys


@pytest.mark.django_db
def test_run_instance_hands_an_empty_group_for_an_instance_without_documents() -> None:
    # run_instance() prunes nothing: replacing the instance's documents with
    # none is the only way to delete what the output still holds for it.
    empty = Category.objects.create(name="")
    rag.register(Category, fields=["name"])

    output = RecordingOutput()
    SyncPipeline(output).run_instance(empty)

    assert output.replaced == [{f"testapp.category:{empty.pk}": []}]


@pytest.mark.django_db
def test_pipeline_prunes_a_model_without_instances_keeping_no_key() -> None:
    # No instance left: every document the output still holds for the model
    # is stale, and only prune can delete them.
    rag.register(Category, fields=["name"])

    output = RecordingOutput()
    SyncPipeline(output).run()

    assert output.replaced == []
    assert output.pruned == [("testapp.category", set())]


@pytest.mark.django_db
def test_pipeline_prunes_a_model_after_its_documents_keeping_their_keys() -> None:
    # The instance without a document is not kept: whatever the output still
    # holds for it is stale, and the prune deletes it.
    lamps = Category.objects.create(name="Lamps")
    empty = Category.objects.create(name="")
    lighting = Category.objects.create(name="Lighting")
    rag.register(Category, fields=["name"])

    output = RecordingOutput()
    SyncPipeline(output).run()

    assert output.pruned == [
        (
            "testapp.category",
            {f"testapp.category:{lamps.pk}", f"testapp.category:{lighting.pk}"},
        )
    ]
    assert f"testapp.category:{empty.pk}" not in output.pruned[0][1]
    # A prune before a replace would delete documents the replace then
    # brings back, or keep a key the replace has not stored yet.
    assert "replace" in output.calls
    assert output.calls[-1] == "prune"


@pytest.mark.django_db
def test_pipeline_prunes_each_model_it_runs_in_order_under_its_own_label() -> None:
    # Page is registered before Category, though declared after it and after
    # it alphabetically: only the order the models run in can put it first.
    faq = Page.objects.create(title="FAQ", slug="faq")
    lighting = Category.objects.create(name="Lighting")
    rag.register(Page, fields=["title"])
    rag.register(Category, fields=["name"])

    output = RecordingOutput()
    SyncPipeline(output).run()

    # One prune per model, each naming its own model with only its own keys:
    # a single prune, or a label shared across models, would let one model's
    # prune delete the documents of another.
    assert output.pruned == [
        ("testapp.page", {f"testapp.page:{faq.pk}"}),
        ("testapp.category", {f"testapp.category:{lighting.pk}"}),
    ]


@pytest.mark.django_db
def test_pipeline_hands_a_model_groups_in_one_batch_per_chunk_of_instances() -> None:
    # One more instance than a chunk holds: one replace per instance, or a
    # single replace for the whole model, would both miss the two batches.
    Category.objects.bulk_create(
        Category(name=f"Category {number}") for number in range(1001)
    )
    rag.register(Category, fields=["name"])
    keys = [
        f"testapp.category:{pk}"
        for pk in Category.objects.order_by("pk").values_list("pk", flat=True)
    ]

    output = RecordingOutput()
    SyncPipeline(output).run()

    assert [list(groups) for groups in output.replaced] == [keys[:1000], keys[1000:]]


class ExtractionFailedError(Exception):
    """Raised by an extractor in the middle of a run."""


@pytest.mark.django_db
def test_pipeline_does_not_prune_a_model_whose_extractor_raises() -> None:
    # The failure comes after the first instance: a prune at that point would
    # keep only the keys extracted so far and delete every other document the
    # output holds for the model.
    Category.objects.create(name="Lamps")
    broken = Category.objects.create(name="Broken")

    @rag.register_extractor(Category)
    class FailingCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            if instance.pk == broken.pk:
                raise ExtractionFailedError(instance.name)
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    with pytest.raises(ExtractionFailedError, match="Broken"):
        SyncPipeline(output).run()

    assert output.pruned == []
    assert "prune" not in output.calls


@pytest.mark.django_db
def test_run_instance_hands_nothing_to_its_output_when_the_extractor_raises() -> None:
    # An empty group for the failed instance would delete the documents the
    # output still holds for it, as if it had none.
    broken = Category.objects.create(name="Broken")

    @rag.register_extractor(Category)
    class FailingCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            raise ExtractionFailedError(instance.name)

    output = RecordingOutput()
    with pytest.raises(ExtractionFailedError, match="Broken"):
        SyncPipeline(output).run_instance(broken)

    assert output.calls == []


@pytest.mark.django_db
def test_pipeline_hands_over_each_chunk_before_extracting_the_next() -> None:
    # The failure is on the first instance of the second chunk: a pipeline
    # extracting the whole model before handing anything over would lose the
    # 1000 documents already extracted.
    Category.objects.bulk_create(
        Category(name=f"Category {number}") for number in range(1001)
    )
    pks = list(Category.objects.order_by("pk").values_list("pk", flat=True))
    keys = [f"testapp.category:{pk}" for pk in pks]

    @rag.register_extractor(Category)
    class FailingLastCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            if instance.pk == pks[-1]:
                raise ExtractionFailedError(instance.name)
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    with pytest.raises(ExtractionFailedError, match="Category 1000"):
        SyncPipeline(output).run()

    assert [list(groups) for groups in output.replaced] == [keys[:1000]]
