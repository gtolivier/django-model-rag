# Roadmap

How django-model-rag gets built. Each feature below is one branch and one
pull request, built test-first with the `/tdd:feature` workflow (see
[AGENTS.md](AGENTS.md)). The detailed list of behaviors of a feature is
agreed when that feature starts, not here. The pull request that completes a
feature also ticks it here.

## Decisions

- **No chunking in this package.** Chunking, embedding and vector storage
  belong to
  [django-minimal-rag](https://github.com/gtolivier/django-minimal-rag). The
  earlier prototype had a naive chunker in its pipeline; it is not
  rewritten. The pipeline hands the `NormalizedDocument`s to an output that
  the project supplies (its shape is an open question, below).
- **The interface between the two packages is structural.**
  django-minimal-rag will define a `typing.Protocol` that describes what it
  reads from a document (`text`, `title`, `url`, `source_key`…), with
  read-only members (`@property`), which a dataclass field and a property
  both satisfy. `NormalizedDocument` will satisfy it by its shape alone:
  neither package imports the other, and
  [django-model-rag-demo](https://github.com/gtolivier/django-model-rag-demo)
  will type-check the two together — it gets a mypy check when the Protocol
  exists. The attribute names of `NormalizedDocument` are therefore a
  contract as soon as they exist; the exact contents of the Protocol are
  settled when django-minimal-rag is designed.

## Rewrite of the prototype

The prototype is the behavioral reference: each behavior is re-derived from
a failing test, and its code is not copied. Leaves come first, so that each
feature builds on code that is already green.

- [ ] **1. `guess_text_fields(instance)`** — which fields of a model hold
  semantic text: text field types, excluded names, text fields whose name
  ends in `_id`, and an order that puts title-like names first.
- [ ] **2. `guess_language(instance)` and `guess_url(instance)`** — a
  language read from common attribute names, a URL from
  `get_absolute_url`, and their fallbacks.
- [ ] **3. `resolve_relation_text(instance, relation_name)`** — the text of
  related objects, across foreign keys, reverse relations and many-to-many
  relations; a missing or empty relation yields no text.
- [ ] **4. `NormalizedDocument`** — its attributes, a stable `source_key`
  (`app_label.model_name:pk`) and a readable `repr`.
- [ ] **5. `BaseExtractor` and `GenericModelExtractor`** — declared or
  guessed fields, followed relations, no document for empty content, a
  title taken from the first field (or the instance's string form when there
  is no field), a language and a URL from configured fields or from the
  guesses, permissions passed through.
- [ ] **6. The registry and the `rag` singleton** — `register`,
  `register_extractor` (a class decorator), `registered_models`,
  `extractor_for`, `is_registered`, and the public API: `rag`,
  `BaseExtractor`, `NormalizedDocument` and `SyncPipeline` importable from
  `django_model_rag`, as a project needs them to write its own extractors.
- [ ] **7. `SyncPipeline`, without a chunker** — a full run over the
  registered models or a subset of them, a run for a single instance, and
  extractors that return one document, several or none; each document goes
  to the output. The questions below are settled before it starts.

Then, in django-model-rag-demo, an integration test replays the prototype's
demo scenario against the installed package: a product with its category, a
product without a description, and a page whose content is spread across
plugin models.

### Open questions, before feature 7

These decide the interface that the command, the signals and
django-minimal-rag all depend on.

- **The shape of the output.** A single callable taking a document cannot
  later remove anything without an API break. A small object (a Protocol
  with, say, an upsert and a delete) avoids it.
- **The identity of a document.** An instance can produce several
  documents, and they share one `source_key`. Either the key identifies the
  group — the documents of an instance are replaced together — or each
  document gets its own part.
- **Removal on save.** A saved instance that now produces fewer documents,
  or none, must have the old ones removed, not only a deleted instance.
- **Where the output comes from.** The command and the signals run outside
  project code, so they need a configured destination (a setting, for
  instance).

## Synchronization

The prototype has none of this: the behaviors come from design, not from a
reference.

- [ ] **8. A management command, `sync_model_rag`**, that runs the pipeline
  over every registered model.
- [ ] **9. Signals** — `post_save` re-extracts the saved instance;
  `post_delete` removes its documents. Removal is new: the output needs a
  way to delete by `source_key`, designed together with
  django-minimal-rag's handling of updated and orphaned chunks. Open: text
  that comes from another model — through `follow`, or a parent that a
  custom extractor reads — goes stale when that model changes, unless the
  dependent instances are found and re-extracted.

## Not planned here

- **django-minimal-rag** gets its own roadmap after a design pass
  (embeddings, LLM, permission filtering, orphaned chunks). It has no
  prototype to rewrite.
- **Adapters for django CMS and Wagtail** will be separate packages, later.
