from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

import pytest
from django.db.models import QuerySet

from django_model_rag import (
    BaseExtractor,
    DocumentOutput,
    NormalizedDocument,
    SyncPipeline,
    rag,
)
from tests.recording import RecordingOutput
from tests.testapp.models import AccordionItem, Category, Page, Product


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

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        pass


def test_a_class_whose_replace_takes_a_flat_sequence_is_not_a_document_output() -> None:
    # The type checker must reject the assignment: replace receives the
    # documents grouped by source key, not one after the other. If the
    # protocol ever accepted a flat sequence, warn_unused_ignores would flag
    # this line.
    output: DocumentOutput = FlatReplaceOutput()  # type: ignore[assignment]

    assert hasattr(output, "prune")


class MutableSetPruneOutput:
    """An output whose prune requires the kept keys as a mutable set."""

    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        pass

    def prune(self, model_label: str, kept_keys: set[str]) -> None:
        pass


def test_a_class_whose_prune_takes_a_mutable_set_is_not_a_document_output() -> None:
    # The type checker must reject the assignment: prune receives the kept
    # keys read-only, so an output may not rely on mutating them, and the
    # pipeline may hand a frozenset. If the protocol ever required a mutable
    # set, warn_unused_ignores would flag this line.
    output: DocumentOutput = MutableSetPruneOutput()  # type: ignore[assignment]

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

    run_groups = run_output.received_groups()
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

    received_keys = set(output.received_groups())
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
def test_run_instance_hands_an_empty_group_for_an_instance_its_queryset_omits() -> None:
    # run() loads instances through get_queryset(): a draft it leaves out must
    # not get indexed by run_instance() either, and the empty group deletes
    # whatever the output still holds for it.
    draft = Category.objects.create(name="Draft")
    extracted: list[Category] = []

    @rag.register_extractor(Category)
    class PublishedCategoryExtractor(BaseExtractor[Category]):
        def get_queryset(self, queryset: QuerySet[Category]) -> QuerySet[Category]:
            return queryset.exclude(name="Draft")

        def extract(self, instance: Category) -> NormalizedDocument:
            extracted.append(instance)
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    SyncPipeline(output).run_instance(draft)

    assert output.replaced == [{f"testapp.category:{draft.pk}": []}]
    assert extracted == []


@pytest.mark.django_db
def test_run_instance_hands_the_documents_of_an_instance_its_queryset_keeps() -> None:
    # The draft is saved too: a filter that let the hook's exclusion reach
    # every instance, or tested the wrong one, would empty lighting's group.
    Category.objects.create(name="Draft")
    lighting = Category.objects.create(name="Lighting")

    @rag.register_extractor(Category)
    class PublishedCategoryExtractor(BaseExtractor[Category]):
        def get_queryset(self, queryset: QuerySet[Category]) -> QuerySet[Category]:
            return queryset.exclude(name="Draft")

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    SyncPipeline(output).run_instance(lighting)

    assert output.replaced == [
        {
            f"testapp.category:{lighting.pk}": [
                NormalizedDocument(
                    text="Lighting",
                    source_app_label="testapp",
                    source_model="category",
                    source_pk=lighting.pk,
                ),
            ],
        }
    ]


@pytest.mark.django_db
def test_pipeline_hands_once_the_documents_of_an_instance_a_join_repeats() -> None:
    # Filtering across the reverse foreign key without distinct() yields
    # lighting once per matching product: a pipeline extracting every row
    # would hand its document twice in its group.
    lighting = Category.objects.create(name="Lighting")
    Product.objects.create(
        name="Desk lamp", description="A lamp.", price="20.00", category=lighting
    )
    Product.objects.create(
        name="Floor lamp", description="A tall lamp.", price="50.00", category=lighting
    )

    @rag.register_extractor(Category)
    class LampCategoryExtractor(BaseExtractor[Category]):
        def get_queryset(self, queryset: QuerySet[Category]) -> QuerySet[Category]:
            return queryset.filter(products__name__in=["Desk lamp", "Floor lamp"])

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    SyncPipeline(output).run()

    assert output.received_groups() == {
        f"testapp.category:{lighting.pk}": [
            NormalizedDocument(
                text="Lighting",
                source_app_label="testapp",
                source_model="category",
                source_pk=lighting.pk,
            ),
        ],
    }


@pytest.mark.django_db
def test_run_instance_rejects_a_get_queryset_that_returns_no_queryset() -> None:
    # Telling whether the hook keeps the instance must not swallow a broken
    # hook: an empty group would delete the documents the output still holds
    # for lighting, as if get_queryset() had filtered it out.
    lighting = Category.objects.create(name="Lighting")

    @rag.register_extractor(Category)
    class ListingCategoryExtractor(BaseExtractor[Category]):
        # A list of the instances instead of a queryset is the slip under
        # test: the type checker rightly rejects it.
        def get_queryset(  # type: ignore[override]
            self, queryset: QuerySet[Category]
        ) -> list[Category]:
            return list(queryset)

        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    with pytest.raises(
        TypeError, match=r"ListingCategoryExtractor\.get_queryset\(\).*QuerySet"
    ):
        SyncPipeline(output).run_instance(lighting)

    assert output.calls == []


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


def test_run_instance_rejects_an_unsaved_instance_before_extracting_it() -> None:
    # Without a primary key the source key would be "testapp.category:None":
    # a group under it would store documents no saved instance owns, and no
    # later run could replace or prune them under the instance's real key.
    unsaved = Category(name="Lamps")
    extracted: list[Category] = []

    @rag.register_extractor(Category)
    class RecordingCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            extracted.append(instance)
            return self.build_document(instance, text=instance.name)

    output = RecordingOutput()
    with pytest.raises(ValueError, match="primary key"):
        SyncPipeline(output).run_instance(unsaved)

    assert extracted == []
    assert output.calls == []


@pytest.mark.django_db
def test_run_instance_rejects_a_document_whose_source_is_another_instance() -> None:
    # A document attributed to lamps would make the output replace lamps'
    # documents while syncing lighting: each instance's documents must come
    # from that instance.
    lamps = Category.objects.create(name="Lamps")
    lighting = Category.objects.create(name="Lighting")

    @rag.register_extractor(Category)
    class MisattributingCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return self.build_document(lamps, text=instance.name)

    output = RecordingOutput()
    with pytest.raises(TypeError, match="MisattributingCategoryExtractor"):
        SyncPipeline(output).run_instance(lighting)

    assert output.calls == []


@pytest.mark.django_db
def test_pipeline_rejects_a_document_whose_source_is_another_model() -> None:
    # A document attributed to the FAQ page would make the output replace the
    # page's documents while running Category, outside the model being run and
    # pruned: each instance's documents must come from that instance.
    faq = Page.objects.create(title="FAQ", slug="faq")
    Category.objects.create(name="Lighting")

    @rag.register_extractor(Category)
    class MisattributingCategoryExtractor(BaseExtractor[Category]):
        def extract(self, instance: Category) -> NormalizedDocument:
            return NormalizedDocument(
                text=instance.name,
                source_app_label="testapp",
                source_model="page",
                source_pk=faq.pk,
            )

    output = RecordingOutput()
    with pytest.raises(TypeError, match="MisattributingCategoryExtractor"):
        SyncPipeline(output).run()

    received_keys = set(output.received_groups())
    assert f"testapp.page:{faq.pk}" not in received_keys


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
