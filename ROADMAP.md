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
  the project supplies (see "The output", below).
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
- **Each package works without the other.** django-model-rag hands its
  documents to whatever output a project supplies: django-minimal-rag is one
  of them, not a requirement. django-minimal-rag indexes any document of the
  right shape, whatever produced it. Its indexing API is therefore the
  output's API (below), usable directly, not an adapter for this package.

### The output

Settled before feature 8 (they were open questions until then). The output
is the interface that the command (feature 9), the signals (feature 10) and
django-minimal-rag all depend on: adding a method to a Protocol later breaks
every class that implements it, so the output covers from the start what
features 9 and 10 need.

- **A Protocol, defined here.** django-model-rag calls the output, so it
  defines its `typing.Protocol`; django-minimal-rag satisfies it by its
  shape, without importing it — the document's interface, the other way
  round. A single callable taking documents was rejected: it could never
  remove anything without an API break. So was a per-document `upsert` /
  `delete` pair: a save that produces fewer documents would need two calls,
  with a moment where the instance has none, and `upsert` would not say
  whether it adds to an instance's documents or replaces them.
- **Two methods, by source.** `replace(groups)` takes a mapping of
  `source_key` to the complete sequence of that source's documents: each
  group replaces everything the output holds for its source, and an empty
  sequence removes the source. `prune(model_label, kept_keys)` removes the
  sources of a model (`app_label.model_name`: the whole part of their
  `source_key` before the colon, so that `app.note` never matches
  `app.notebook:3`) that are not in `kept_keys`. Groups go in batches, so that
  django-minimal-rag can batch its embedding calls. The exact signatures are
  settled by the tests of feature 8; the name of `prune`'s first argument
  should stay meaningful for a django-minimal-rag used without this package.
- **`source_key` identifies a group, not a document.** The documents of an
  instance are replaced together; no document gets an identity of its own
  (a rank shifts on every insertion, a name would burden every extractor).
  django-minimal-rag identifies its chunks within a group, and can compare
  texts to re-embed only what changed.
- **A document's source is the instance it was extracted from.** A custom
  extractor that builds a document whose source is another instance — a
  plugin indexed as its page — would replace that other instance's
  documents, and its own deletion would remove nothing. The pipeline refuses
  it. Content gathered from other models is indexed by an extractor on the
  model that owns it (a `Page` extractor reading its plugins); keeping it
  fresh when those models change is feature 10's open question.
- **Removal is replacement.** Every run of an instance sends its complete
  set of documents, empty included: an extractor returning `None` means
  "nothing to index any more", hence removal. An extractor that raises sends
  nothing, and the output keeps the previous documents. django-model-rag
  stores nothing about what it produced — remembering it would make the
  package a Django app with models and migrations, duplicating
  django-minimal-rag's storage. Leaving stale documents to the next full
  sync was rejected too: on an authenticated intranet, an unpublished page
  would stay quotable until then.
- **The retry is running again.** Every call replaces whole groups, so
  replaying it gives the same result. A `run()` interrupted in a model
  leaves the batches it sent up to date, the instances it had not reached
  with their previous documents — stale, not missing — and that model
  unpruned, since an incomplete list of kept keys would delete live
  sources; the models after it are not run. The output never loses a
  source that still exists, and running `run()` again (or `run([Model])`)
  repairs the rest. A `run_instance()` that fails sends nothing: the
  instance keeps its previous documents until its next save or the next
  full run. What this leaves to the other layers: the atomicity of a batch
  is the output's (django-minimal-rag applies each `replace()` in a
  transaction), and so are transient errors such as a rate-limited
  embedding API, retried with backoff — the pipeline cannot tell which
  exceptions are transient. There is no checkpoint, so a run again
  re-extracts everything; django-minimal-rag compares texts and re-embeds
  only what changed.
- **The destination is a setting, read by the command and the signals.**
  `SyncPipeline` receives its output explicitly, so tests and scripts pass
  their own. The command and the signals run outside project code: they
  build the output from a setting shaped like Django's `STORAGES` — a
  dotted path to a class and its options, for instance
  `{"BACKEND": "…", "OPTIONS": {…}}` — loaded with `import_string`. Without
  the setting, the command fails with `ImproperlyConfigured`. The setting's
  name and loading come with feature 9, their first user. Registering the
  output in code (`rag.set_output()` from `AppConfig.ready()`) was rejected:
  global mutable state, dependent on app order, and a per-environment value
  belongs in settings.
- **No project-wide default language, for now.** A model with no language
  source keeps giving `None`, which tells the output the language is
  unknown; the output decides what to make of it (django-minimal-rag may
  fall back on `LANGUAGE_CODE`). Falling back on `LANGUAGE_CODE` here would
  be wrong for a multilingual site; an opt-in `DEFAULT_LANGUAGE` would
  reopen feature 7's rule that a configured field is authoritative. Adding a
  default later breaks nothing; removing one would.
- **No project-wide defaults for guessed fields, for now.** `exclude=` per
  model and the built-in title-like names stay. A setting excluding names
  (`password`, `token`…) contradicts feature 4 — no field is left out by its
  name — gives false safety (`secret_answer` passes), and changing it would
  silently change what every model indexes. A later, non-breaking
  possibility: a list of sensitive names that only warns, as a system check
  next to feature 4's.

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
- [x] **7. Language, URL and permissions** — each document's language
  comes from a constant `language=`, a `language_field` (an own field or a
  lookup path), or an own field guessed by name (`language`, then
  `language_code`, then `lang`), else `None`; its URL from a `url_field`,
  else `get_absolute_url()` (left relative, its exceptions propagated), else
  `""`. A configured or guessed field is authoritative: blank gives `None`
  or `""`, never the next source. `permissions=["app_label.codename", ...]`
  puts a `frozenset` on every document of the model, and custom extractors
  pass their own through `build_document`.
- [x] **7b. The shape of the queryset** — `BaseExtractor.get_queryset()`
  shapes the queryset the instances are loaded from (`select_related`,
  `prefetch_related`…); it must return a `QuerySet`, and the pipeline
  iterates it with `iterator(chunk_size=1000)`. A single-instance run only
  asks it whether it keeps the instance (feature 8). The extractor built
  by `register()` uses it to load only the columns it reads: its own
  declared fields and single-field options, the related columns its lookup
  paths name, and the text columns of followed relations with the fields
  that link them back. Every own column stays loaded when
  `get_absolute_url()` or `str(instance)` may read any of them. The
  pipeline no longer duck-types extractors, and the single-field options
  (`title_field`, `language_field`, `url_field`) are listed once. Two
  cases are left open, with no test:
  - `get_queryset()` must return a queryset of the registered model itself,
    so the queryset of one of its proxy models is refused — although it
    holds the same rows, under a class whose methods (`__str__`,
    `get_absolute_url()`) may differ. Decide whether to accept it.
  - A followed model whose default manager makes a nested
    `select_related()` (`product__category`, two links deep) goes through
    the prefetch that loads only its text columns. A quick check on the test
    bench (`Offer`, followed from `Product`) ran in two queries with the
    right text, but no test pins it down.
- [x] **8. The output** — `SyncPipeline(output=...)` hands the documents to
  an output that the project supplies, as decided under "The output", above:
  - the output's Protocol, importable from `django_model_rag`, with
    `replace(groups)` and `prune(model_label, kept_keys)`, typed against
    `NormalizedDocument`;
  - `run()` sends the documents of each model in batches of groups, one
    batch per chunk of the iterator, then prunes the model with the keys of
    the sources that produced documents — so an instance that now produces
    none is removed. A model whose run fails is not pruned;
  - `run_instance()` sends its instance's group, even empty;
  - a document whose source is not the instance it was extracted from fails
    with a `TypeError` naming the extractor;
  - the output is a required argument of `SyncPipeline`, and `run()` and
    `run_instance()` return `None`: the documents go only to the output, so
    a large run holds no list of them. Tests read them back through an
    output that records what it receives.

Then, in django-model-rag-demo, an integration test replays the prototype's
demo scenario against the installed package: a product with its category, a
product without a description, and a page whose content is spread across
plugin models.

## Synchronization

The prototype has none of this: the behaviors come from design, not from a
reference.

- [ ] **9. A management command, `sync_model_rag`**, that runs the pipeline
  over every registered model, into the output named by a setting (its
  name and loading come with this feature; see "The output", above). Each
  model's run ends with a `prune`, which removes what the signals missed: an
  instance deleted by raw SQL, with the signals off, or before the package
  was installed. Open: the documents of a model that is no longer
  registered are pruned by no run. Open: a default manager that returns
  subclass instances (django-polymorphic, `InheritanceManager`) keys their
  documents under the subclass's label, while the model is pruned under its
  own, so they are pruned by no run. Open: when one model's extractor
  raises, the exception stops the run; should the command go on with the
  other models and report the failure at the end? Open, to settle by this feature at the
  latest: does the package become a Django app that autodiscovers each
  app's `rag.py`, as `django.contrib.admin` does with `admin.py`? The
  command only sees the models registered by the time it runs; until then,
  each project imports its `rag.py` from `AppConfig.ready()`.
- [ ] **10. Signals** — `post_save` re-extracts the saved instance and
  replaces its group, even empty; `post_delete` replaces it with an empty
  group. Both build their output from the same setting as the command.
  Open: whether they write at once or after the transaction commits
  (`transaction.on_commit`). Open: text
  that comes from another model — through `follow`, or a parent that a
  custom extractor reads — goes stale when that model changes, unless the
  dependent instances are found and re-extracted. Also open: a proxy model
  or a multi-table child of a registered model sends its own class as the
  signal's sender, and `run_instance` looks the exact class up, so such an
  instance is not registered today. Open: a signal that fires while the
  command runs — the run's final `prune` deletes an instance created
  since the run read the table, and a chunk read before an update puts the
  old text back. Open: a failed signal sends nothing; whether
  to retry it through a queue (Celery…) is probably the project's call.

## Not planned here

- **django-minimal-rag** gets its own roadmap after a design pass
  (embeddings, LLM, permission filtering, orphaned chunks). It has no
  prototype to rewrite.
- **Adapters for django CMS and Wagtail** will be separate packages, later.
