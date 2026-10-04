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
a failing test, and its code is not copied. The work goes outside in: each
feature starts from the public API — `rag.register` and `SyncPipeline` — and
the internal functions appear in the refactoring, so that the package does
not inherit the prototype's breakdown.

When a feature starts, its list of behaviors says, for each behavior of the
prototype it covers, whether it is kept, fixed or dropped — broad
`except Exception` clauses, for instance, or a title taken from a field even
when that field is empty. The prototype is a reference to judge, not a model
to copy.

Features are described by what they do. Only the public API is named:
internal functions get their names and signatures from the tests and the
refactoring, not from the prototype.

- [x] **1. `NormalizedDocument`** — its attributes, a stable `source_key`
  (`app_label.model_name:pk`) and a readable `repr`.
- [x] **2. Extract declared fields** — `rag.register(Model, fields=[...])`
  then `SyncPipeline().run()` returns one document per instance of the
  registered models: the text of the declared fields, a title taken from
  `title_field` (the first declared field when it is not given), the source
  of the document, and no document for empty content. `rag.unregister`, and
  `AlreadyRegistered` / `NotRegistered` as in `django.contrib.admin`.
  `fields` is a list or a tuple; an unknown field, a relation, a field
  declared twice or an empty list fails at registration, and so does a
  wrong `title_field`. Values are stripped, a field with choices gives its
  label. `rag`, `NormalizedDocument`, `SyncPipeline` and the two exceptions
  are importable from `django_model_rag`.
- [x] **3. Custom extractors** — `rag.register_extractor` (a class
  decorator) and `BaseExtractor`, also importable from `django_model_rag`;
  extractors that return one document, several or none; a run over a subset
  of the models, a run for a single instance, and the list of registered
  models.
- [x] **4. Guessed fields** — `rag.register(Model)` without `fields`
  extracts the model's text fields, guessed once at registration from their
  type alone: a `CharField` (exactly, not a subclass such as a slug, an
  e-mail or a URL) or a `TextField` (subclasses included). Title-like names
  come first (`title`, `name`, `heading`, `label`), then the others in
  declaration order. `exclude=[...]` leaves guessed fields out; it cannot be
  combined with `fields`. A model with nothing to guess, or whose `exclude`
  leaves nothing, fails at registration. No field is left out by its name:
  a sensitive text field is the project's to exclude. Later, as a Django
  system check (a warning, not an error): `exclude` naming a field that is
  not guessed, which excludes nothing — a CharField subclass, say, that the
  project believes it is leaving out.
- [x] **5. Followed relations** — `follow=[...]` adds the guessed text of
  related objects, one level deep, across foreign keys, one-to-one, reverse,
  many-to-many and generic relations (a generic foreign key is refused),
  named by their accessors as in `prefetch_related`; a missing or empty
  relation adds nothing. The
  instance's string form becomes the title when the model has no field of
  its own to take it from, which only `follow` makes possible. Errors are
  raised at registration; following needs every model loaded, so such
  registrations go in a `rag.py` imported from `AppConfig.ready()`.
  `run()` reads the relations with `select_related` / `prefetch_related`,
  in a fixed number of queries.
- [x] **6. Lookup paths in `fields`** — `fields=["name", "category__name"]`,
  as in `list_display` or `values()`: one field of a related object, in a
  chosen order, where `follow=` takes all its text — and a way to leave out
  a related object's sensitive field, such as `User.password`, which
  `follow=` brings along. The two coexist. Links go through foreign keys and
  one-to-one relations, reverse ones by their query name; `title_field` can
  be a path too. `run()` reads every link in the same query as the
  instances. Open: a path through a reverse foreign key or a many-to-many
  relation (`text_plugins__body`) is refused, since it has no single value;
  `values()` gives one row per related object and `search_fields` searches
  them all. For a document, joining their texts in primary key order would
  be the natural reading — decide whether to allow it, and how it then
  differs from `follow`.
- [ ] **7. Language, URL and permissions** — from configured fields
  (`language_field`, `url_field`) or guessed (common attribute names,
  `get_absolute_url`), with their fallbacks; permissions passed through.
  Once this feature settles everything the pipeline reads from an instance,
  reconsider loading only those columns (`QuerySet.only()`): until then,
  each attribute read outside the loaded ones would cost a query per
  instance. Likewise for custom extractors that read relations: an
  optional hook to shape their queryset (`select_related`,
  `prefetch_related`, iterated with `iterator(chunk_size=...)`) would avoid
  a query per instance.
- [ ] **8. The output** — each document goes to an output that the project
  supplies, instead of only being returned. The questions below are settled
  before it starts.

Then, in django-model-rag-demo, an integration test replays the prototype's
demo scenario against the installed package: a product with its category, a
product without a description, and a page whose content is spread across
plugin models.

### Open questions, before feature 8

These decide the interface that the command, the signals and
django-minimal-rag all depend on.

- **The shape of the output.** A single callable taking a document cannot
  later remove anything without an API break. A small object (a Protocol
  with, say, an upsert and a delete) avoids it.
- **The identity of a document.** An instance can produce several
  documents, and they share one `source_key`. Either the key identifies the
  group — the documents of an instance are replaced together — or each
  document gets its own part. A custom extractor may also build a document
  whose source is another instance (a plugin indexed as its page), which
  nothing forbids yet: decide whether the pipeline should.
- **Removal on save.** A saved instance that now produces fewer documents,
  or none, must have the old ones removed, not only a deleted instance.
- **Where the output comes from.** The command and the signals run outside
  project code, so they need a configured destination (a setting, for
  instance).
- **Project-wide defaults for guessed fields.** The same setting could hold
  names excluded from every guessed model (`password`, `token`…) and the
  title-like names and their order, which `exclude=` and the built-in list
  cover per model today.

## Synchronization

The prototype has none of this: the behaviors come from design, not from a
reference.

- [ ] **9. A management command, `sync_model_rag`**, that runs the pipeline
  over every registered model. Open, to settle by this feature at the
  latest: does the package become a Django app that autodiscovers each
  app's `rag.py`, as `django.contrib.admin` does with `admin.py`? The
  command only sees the models registered by the time it runs; until then,
  each project imports its `rag.py` from `AppConfig.ready()`.
- [ ] **10. Signals** — `post_save` re-extracts the saved instance;
  `post_delete` removes its documents. Removal is new: the output needs a
  way to delete by `source_key`, designed together with
  django-minimal-rag's handling of updated and orphaned chunks. Open: text
  that comes from another model — through `follow`, or a parent that a
  custom extractor reads — goes stale when that model changes, unless the
  dependent instances are found and re-extracted. Also open: a proxy model
  or a multi-table child of a registered model sends its own class as the
  signal's sender, and `run_instance` looks the exact class up, so such an
  instance is not registered today.

## Not planned here

- **django-minimal-rag** gets its own roadmap after a design pass
  (embeddings, LLM, permission filtering, orphaned chunks). It has no
  prototype to rewrite.
- **Adapters for django CMS and Wagtail** will be separate packages, later.
