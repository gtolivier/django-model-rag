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
  fresh when those models change is still open (see feature 10).
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
  the setting, the command fails with `ImproperlyConfigured`. Feature 9
  named it `MODEL_RAG_OUTPUT`: a dict with `BACKEND`, and `OPTIONS` passed
  as keyword arguments to a new instance on each use. Registering the
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
    none is removed. A model whose run fails is not pruned. (Revised by
    10d: `run()` sends an empty group for an instance without documents,
    and the prune keeps its key.);
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

- [x] **9. A management command, `sync_model_rag`**, that runs the pipeline
  over every registered model, in registration order, or over the models it
  is given as `app_label.model_name`, into the output built from
  `MODEL_RAG_OUTPUT`. Each model's run ends with a `prune`, which removes what
  the signals missed: an instance deleted by raw SQL, with the signals off,
  or before the package was installed. Everything is checked before any
  model runs: the setting (`ImproperlyConfigured`, including a backend
  without a callable `replace` or `prune`) and the labels (`CommandError`
  for a label naming no model, or an unregistered one). When a model fails,
  the command writes it and its error on stderr, does not prune it, goes on
  with the others, and ends with a `CommandError` naming every failed model;
  each model that succeeds writes a `synced` line, unless `--verbosity 0`;
  `--traceback` adds each failure's traceback. A model named twice runs
  once; with no model registered, the command warns and ends without
  error. The package became a
  Django app that autodiscovers each installed app's `model_rag.py` in its
  `AppConfig.ready()`, in `INSTALLED_APPS` order, as `django.contrib.admin`
  does with `admin.py`.
  Registering by hand stays: a project changes a third-party app's
  registration with `unregister` and `register` from a later app.
  `ConsoleOutput` (in `django_model_rag.output`) writes what it receives to
  a stream, to try the command without a real output. Still open:
  - the documents of a model that is no longer registered are pruned by no
    run;
  - a default manager that returns subclass instances (django-polymorphic,
    `InheritanceManager`) keys their documents under the subclass's label,
    while the model is pruned under its own, so they are pruned by no run;
  - the command checks `replace` and `prune` on the backend class, so a
    class that only sets them on its instances is refused;
  - `get_absolute_url()` and `str(instance)` may read related objects that
    the queryset does not join — one query per instance on a large table.
    An option could add `select_related` without a custom extractor;
  - progress: a model's `synced` line comes only at its end, nothing per
    chunk on a large table;
  - run inside an outer transaction (`call_command` from code under
    `atomic()`), a database error in one model aborts that transaction,
    and every later model fails too, each blamed on itself. A savepoint
    per model would isolate them, but would keep a transaction open across
    the output's calls, which may be slow;
  - `prune()` receives every kept key of a model at once, which a very
    large table makes a large set;
  - a first sync of a large site calls the output for every instance: the
    embedding API's quota and cost are the output's to manage
    (django-minimal-rag). Reading from a replica is untested: a custom
    extractor's `get_queryset()` could route it with `using()`.
- [x] **10. Signals** — `post_save` re-extracts the saved instance and
  replaces its group, even empty; `post_delete` replaces it with an empty
  group. Both build a new output from `MODEL_RAG_OUTPUT` each time, as the
  command does. Decided in this feature:
  - **After the commit.** The signal keeps only the model and the primary
    key; `transaction.on_commit` reloads the instance and extracts it then,
    so the output sees only committed data, including the inlines and
    many-to-many relations an admin form saves after `post_save`. An
    instance saved then deleted before the commit gets only the delete's
    empty group. A rolled-back transaction sends nothing.
  - **Synchronously**, in the process that commits: extraction costs a few
    queries, and the slow part (embedding) is the output's, which may queue
    its own work.
  - **A failure is logged**, not raised: an extractor, an output or a
    database error reloading the instance that raises in a commit callback
    is logged on the `django_model_rag` logger with the source key, and the
    other callbacks still run. The instance keeps its previous documents
    until its next save or the next `sync_model_rag`. A missing or invalid
    `MODEL_RAG_OUTPUT` — `OPTIONS` its `BACKEND` does not accept included —
    is the exception: it raises `ImproperlyConfigured` at the save (in
    `pre_save`, so no row is written, even in autocommit) or the delete
    (rolled back), so a forgotten setting never silently stops the indexing.
  - **A delete sends its empty group directly**, without the extractor: the
    row is gone, so there is nothing for `get_queryset` to filter.
  - **Proxies and multi-table children.** Saving or deleting through a
    proxy of a registered model syncs the registered model's group, under
    its label. Saving a multi-table child syncs each registered model among
    the child and its parents, each under its own label and its own primary
    key (a child may declare a primary key of its own next to its
    `parent_link`); deleting it empties the nearest registered one's group.
    Unregistering a proxy keeps the delete listener its registered concrete
    model needs.
  - **Off switch:** `MODEL_RAG_SIGNALS = False`. A raw save (`loaddata`) is
    ignored; the command catches up.
  - **Fast delete kept.** `post_delete` is connected per registered model
    (and its proxies), never globally, so unregistered models keep Django's
    fast delete.

  Still open:
  - text that comes from another model — through `follow`, a lookup path,
    or a parent that a custom extractor reads — goes stale when that model
    changes, unless the dependent instances are found and re-extracted.
    Reverse relations in `follow` are done (11a), forward foreign keys and
    one-to-ones in `follow` and in lookup paths too (11b), a custom
    extractor's dependencies too (11c); many-to-many relations (11d) are
    not. Until then, a project connects its own receiver that runs
    `run_queryset` on the dependent instances, and the command repairs the
    rest;
  - background tasks: a slow output delays the response that saves the
    instance. A task — Django's `django.tasks` (6.0+; a separate package
    on 5.2), or a queue on Redis or RabbitMQ — would receive the model
    label and the primary key, as the commit callback already does, so the
    output would need nothing new;
  - retrying a failed signal: the output's job (transient errors) or a
    queue's, probably the project's call;
  - a context manager pausing the signals for a bulk import followed by a
    sync (`with rag.signals_paused():`), on top of the setting (12a);
  - batching: each saved or deleted instance gets its own commit callback,
    which builds its own output and makes its own `replace()` call, so
    deleting a queryset of 10,000 rows makes 10,000 of each, in the
    committing request. Batching them per transaction needs state shared
    by the callbacks of one transaction (12b);
  - a proxy registered instead of its concrete model gets no sync from the
    signals, which look up the concrete model and its parents;
  - a proxy defined after its concrete model is registered gets no
    `post_delete` listener: deleting through it empties nothing. Registering
    from `model_rag.py`, once every model is loaded, avoids it;
  - an instance saved, then deleted by raw SQL in the same transaction,
    gets no empty group: no `post_delete` fires. The command prunes it;
  - saving through a registered multi-table parent does not re-sync a
    registered child, whose documents include the inherited fields: only
    `post_save` with the parent as sender fires. Finding the child row costs
    a query per registered child model on every parent save; it belongs with
    the text that comes from another model, above.
- [x] **10b. Signals during a sync.** A signal that fires while
  `sync_model_rag` runs: the run's final `prune` deleted an instance created
  since the run read the table — a live source lost, which the contract
  promises never happens. Fixed in `run()`: just before each model's
  `prune`, it reads the primary keys that exist again. Decided in this
  feature:
  - **The second read goes through the extractor's `get_queryset()`**, not
    the default manager: an existing instance that `get_queryset()` filters
    out was never read, and must not escape the prune.
  - **Kept keys:** those still there that produced documents, plus those
    still there that the run did not read (revised by 10d: every key still
    there). An instance created during the run is kept (its own signal sent
    its documents); an instance deleted after its chunk was read is pruned,
    though the chunk's `replace()` may have sent its documents back after
    the delete's empty group.
  - **Memory:** the run holds the keys it read and the keys that produced
    documents, and keeps the current keys less those read without
    documents. Tracking only the keys read without documents, chunk by
    chunk, would be cheaper, but it is not the same set when a join in
    `get_queryset()` repeats an instance across two chunks, the first
    producing documents and the second none: it would prune documents the
    output holds. (Gone with 10d: the run holds no keys while it reads.)
  - **One more query** per model run, after every other, streamed with
    `iterator()`.
  - **Transaction isolation** is accepted and documented: under an outer
    `atomic()` in REPEATABLE READ (MySQL's default), the second read sees
    the transaction's snapshot and misses the new instance.
  - **Tests** simulate the signal with an output whose `replace()` creates
    or deletes an instance during the run, with the signals off.

  Still open:
  - a chunk read before an update puts the old text back until the next
    save, a short window;
  - ~~an instance read without documents, then saved during the run so
    that it has some: its signal sends them, and the prune deletes them~~ —
    closed by 10d for a save after its chunk's `replace()`; saved before
    it, the chunk's empty group removes them, the window above;
  - an instance created and committed between the second read and the
    `prune()` call is still deleted, a window of one call;
  - `get_queryset()` runs twice per model run: a filter that depends on
    the time (`published_at__lte=now()`) can keep, as created during the
    run, an instance that became visible without a save, and that the run
    never extracted;
  - with a replica router, the primary keys read again come from the
    replica, which may not have the new instance yet (see 10c).
- [ ] **10c. Signals on several databases** — postponed: no project needs
  several databases yet, and nothing here depends on it. The signals follow
  the default database only: `transaction.on_commit` is attached to it, and
  the commit callback reloads from it, through the database router. With a
  single database, nothing goes wrong. Then, from the most likely case:
  - **A replica router**: the reload reads a lagging replica, which may miss
    a new instance (taken for deleted, so nothing is sent), return the old
    row, or still see a row that `get_queryset()` now keeps out. Stale, not
    corrupt: the next save or `sync_model_rag` repairs it.
  - **A model on another alias** (routed by app, or `using=`): the save or
    the delete waits for the default database's commit and reloads, or
    empties, the group of the default database's row with the same primary
    key — a live document removed until its next save or the next sync.

  Already documented in the README. The design discussed when the feature
  was first started:
  - **Which aliases.** Follow the signal's `using` only when it is the
    database the router writes the model to (`router.db_for_write(model)`,
    without instance hints): the callback is attached to that alias, and
    the instance reloaded from it with `.using(alias)`, not from the
    router's read database. A write elsewhere (a backup or archive copy)
    is ignored, and does not check `MODEL_RAG_OUTPUT`. Without a router,
    only `default` is followed. Rejected: following every alias (a delete
    on a copy empties the live group, and `sync_model_rag` never reads that
    database), ignoring every alias but `default` (a model routed elsewhere
    is never indexed, silently), a setting listing the aliases (one more
    setting, and listing two databases that hold the same model brings back
    the key collision below).
  - **The alias reaches `run_instance()` through the instance**:
    `instance._state.db`, which Django documents. Its `get_queryset()`
    check runs on `.using(instance._state.db)`, and an extractor can read
    `queryset.db`. No public signature changes: a `using=` argument would
    be a second source of truth, and changing `get_queryset()` would break
    the contract of feature 8.
  - **Followers take the alias of the save** (feature 11a): the read of the
    followers a row had before its save, the follower lookup through a
    `to_field`, the follower's commit callback and its reload all use the
    signal's `using`. Django keeps a relation within one database, so the
    follower lives on the same alias as the followed row.
  - **The group key keeps no alias.** Adding it would break `source_key`
    (feature 1) and the split that `prune` makes at the colon, make `run()`
    know the alias, and re-key every index. With one followed database per
    model, two databases share keys only when a router writes the same
    model to several of them without instance hints (tenants chosen by a
    thread-local, say): to be documented as unsupported, and left open.
  - **No `--database` for `sync_model_rag`**: the command already reads the
    router's database, and a `prune` run on another alias would delete the
    default database's sources (the key collision above). Left open.
  - **The test bench**: a second in-memory SQLite alias in
    `tests/settings.py` (pytest-django sets up only the databases a test
    asks for, with `@pytest.mark.django_db(databases=[...])`), commit
    callbacks captured with `django_capture_on_commit_callbacks(using=...)`,
    and two test routers installed through the `settings` fixture: one
    reading and writing the test app's models on the second alias, one
    writing to `default` and reading from the second alias — a replica that
    never receives the writes, hence the worst lag, with an older row put
    there by `bulk_create`, which sends no signal.

  Still open after it: the related objects that `extract()` reads (inlines,
  `follow`, lookup paths on an instance already loaded) go where the router
  sends reads, so a lagging replica can still give them stale text.
- [x] **10d. Instances without documents during a sync.** An instance that
  `run()` read without documents, then saved during the run so that it has
  some: its signal sent them, and the prune deleted them (an open point of
  10b). Keeping every key read without documents would have undone the
  prune of an instance that now produces none. Fixed by moving that
  removal from the prune to `replace()`. Decided in this feature:
  - **`run()` sends an empty group** for every instance it reads that
    produces no documents, in its chunk's `replace()`, as `run_instance()`
    already did: "Removal is replacement", above. This revises feature 8,
    where `run()` sent no group for such an instance and left its removal
    to the prune.
  - **The prune keeps every key** that the extractor's `get_queryset()`
    keeps once the model is run, those read without documents included:
    their empty group already removed what the output held for them. An
    instance saved with documents during the run, once its chunk was
    handed over, keeps what its signal sent. The run no longer holds any
    set of keys while it reads, so 10b's memory trade-off is gone.
  - **Nothing new is asked of an output**: the empty group is the existing
    contract, and django-minimal-rag keeps working without this package.
    The cost is one more group per instance without documents on every
    run, and removing a source the output does not hold must do nothing.
    `ConsoleOutput` writes a `<key> removed` line for each. The README
    suggests filtering such instances out in `get_queryset()` on a large
    table.
  - **An instance a join repeats is handed over once, from its first row**,
    even when its rows straddle two chunks: without it, an empty group in
    the later chunk would delete what the earlier one sent. Rows come in
    primary key order, so an instance's rows are adjacent, and the run only
    remembers the previous key, not a set of every key read.
  - **Tests**: an output whose `replace()` renames, during the run, an
    instance read with an empty name, with the signals off; an output that
    holds what it receives shows the documents of that save survive the
    prune. A join annotating each category with its products' names puts
    one category's rows on both sides of a chunk boundary.
- [x] **10e. `configured_output` in the public API.** The README tells a
  project to call `run_instance()` from a receiver of its own for what the
  signals miss — a related object whose text a registered model reads, a
  `QuerySet.update()` — but the output built from `MODEL_RAG_OUTPUT` was
  only reachable through `django_model_rag.output`. `configured_output()`
  is now importable from `django_model_rag`, as the command and the
  signals use it: same checks, a new output on each call. `OPTIONS` its
  class does not accept now fail with `ImproperlyConfigured`, naming the
  class, before it is built — as the signals already checked, when Python
  can read the class's signature — rather than with the class's own
  `TypeError`. Found by the demo project, whose
  receivers resync the blocks of a saved page.
- [x] **10f. `SyncPipeline.run_queryset(queryset)`.** The demo project's
  receivers resynced the instances related to a saved one by calling
  `run_instance()` once per instance: one `replace()` each. `run_queryset()`
  sends the groups of the instances of a queryset of a registered model in
  batches, like `run()`, without pruning: the model's other instances keep
  their documents. Decided in this feature:
  - **Batches as in `run()`**: one `replace()` per chunk of 1000 instances,
    in primary key order whatever order the queryset sets.
  - **The extractor's `get_queryset()` decides, as in `run()`**: each chunk
    is reloaded through it, so an instance it filters out gets an empty
    group without being extracted, what it adds (an annotation) reaches
    `extract()`, and an instance a join repeats is sent once. An instance
    without documents gets an empty group too.
  - **The caller's queryset names the instances, nothing more**: only its
    primary keys are read, distinct — an instance a join in it repeats is
    extracted and sent once, even across two chunks — and from the
    database it reads from, so `.using()` is honoured.
  - **Errors before anything is sent**, even for an empty queryset: a model
    that is not registered raises `NotRegistered`; a `get_queryset()` that
    returns no `QuerySet` of the model's instances, or a sliced queryset,
    raises `TypeError`. Reordered by primary key, a slice would name other
    instances; the README shows the `pk__in` subquery to use instead.
  - **Not decided here**: a `values()` queryset works, since only its
    primary keys are read, but no test pins it down.
- [ ] **10g. `run_instance()` extracts the instance `get_queryset()`
  loads.** It still extracts the instance it is given, after asking the
  hook's queryset whether it keeps it, so what the hook adds (an
  annotation, a `select_related`) does not reach `extract()` there, while
  `run()` and `run_queryset()` extract the reloaded instance. Aligning it
  costs no extra query: the query asking whether the hook keeps the
  instance can load it.
- [x] **11a. Resync the instances that follow a reverse relation.** Saving
  or deleting an instance that a registered model reaches through a reverse
  foreign key or reverse one-to-one it follows — a text plugin of a page
  registered with `follow=["text_plugins"]` — replaces that registered
  instance's group at the commit, as a save of the instance itself does.
  Requested by the demo project, whose receivers resynced a page when one of
  its blocks was saved. Decided in this feature:
  - **Fan-out 1**: the changed row points to its single parent through the
    foreign key, so finding it costs no query when the key targets the
    parent's primary key — one query for a `to_field`. A forward foreign key
    or a many-to-many reaches many instances: that is 11b and 11d.
  - **Same path as a save**: at the commit, the parent is reloaded,
    extracted and its group replaced; `MODEL_RAG_SIGNALS`, raw saves and a
    missing `MODEL_RAG_OUTPUT` (raised at the save or the delete) behave as
    for a registered model, and a failing extractor is logged under the
    parent's source key.
  - **A move resyncs both parents**: `pre_save` reads the row as committed,
    so that the parent it leaves is replaced too. It does so whenever the
    instance has a primary key, `adding` or not: an instance built with an
    existing key is saved as an UPDATE, and can move too. A model whose
    primary key gets a default (a UUID) pays that read on each insert. A
    save that does not move it replaces its parent once. A model nothing
    follows costs the save no query; a null foreign key schedules nothing.
  - **A delete** resyncs the parent without the deleted row; a parent
    deleted with its children (cascade) gets only its empty group.
  - **Proxies and multi-table children**: a save or a delete through a
    proxy of the followed model resyncs the parent too, and its proxies get
    the `post_delete` listener. A save of a multi-table child of the
    followed model — a row of it too — resyncs the parent as well.
  - **Fast delete**: `post_delete` is connected to the followed model only
    while a registered model follows it; unregistering the last one gives
    its fast delete back. A reverse many-to-many in `follow` connects
    nothing and sends nothing (11d). The reverse of a multi-column
    `ForeignObject`, left out here, is followed since 11c-bis.
  - **Several children of one parent** saved in one transaction replace its
    group once each: batching waits for 12b.
  - **Left for later**, from the review: each save of any model checks, in
    Python, what every registered model follows (an index of the followed
    senders, built at registration, would avoid it); a save whose
    `update_fields` names no followed foreign key still reads the row; a
    cascade schedules one callback per deleted child; a group reached both
    as a registered model and as a follower is replaced twice. None changes
    what is sent; the duplicates belong with the batching of 12b.
- [x] **11b. Resync through forward foreign keys and lookup paths.**
  Saving or deleting an instance that a registered model reaches through a
  forward foreign key or one-to-one — in `follow` (`Product` with
  `follow=["category"]`), or as a link of a lookup path in `fields`,
  `title_field`, `language_field` or `url_field`
  (`"product__category__name"`, however many foreign keys deep) — replaces
  the groups of the registered instances that reach it, at the commit.
  Decided in this feature:
  - **One batch per model**: the followers of a saved or deleted instance
    are sent through `run_queryset`, one commit callback per follower
    model, so a category followed by many products makes one query for
    them, not one each. The batch is cut into queries of 500 rows, under
    the variable limit of the oldest SQLite builds. A follower reached
    through two declarations is sent once. A failing follower stops the
    rest of its batch of 500, whose failure is logged: the project reads
    the cause and fixes it, then the command repairs the groups not sent.
    The other batches still run.
  - **One log message**: a failure is logged as `Syncing <follower model>
    instances that follow <source key> failed`, naming the concrete model
    of a row saved or deleted through a proxy. 11a's message changed to
    match.
  - **Creating costs nothing**: a row just created has no follower
    pointing to it yet, so it looks none up. `pre_save` reads the row as
    committed only for the reverse relations of 11a: a row cannot move
    away from those pointing to it.
  - **Deletes are resolved in `pre_delete`**, while the path still leads
    somewhere: a `SET_NULL` clears the foreign keys before `post_delete`.
    `pre_delete` and `post_delete` are connected together, to the
    followed models (those a deep path reaches included) and their
    proxies, only while a registered model follows them. A follower
    deleted by the cascade gets only its empty group.
  - **Same rules as 11a** for `MODEL_RAG_SIGNALS`, raw saves, a missing
    `MODEL_RAG_OUTPUT` (raised at the save or the delete, writing no row),
    proxies and multi-table children. A foreign key that names a proxy
    (`ForeignKey(CategoryProxy)`) is followed like one that names its
    concrete model, on saves and deletes. A foreign key with a `to_field`
    is matched on that column; a null target value matches no row.
  - **Left for later**: a path is followed only through its leading
    foreign keys: past a reverse one-to-one
    (`supplier__supplier_profile__body`), saving or deleting the profile
    resyncs nothing; a `GenericRelation` in a path is not followed either.
    Changing the `to_field` value of a row whose followers' foreign key has
    `db_constraint=False` is not tested. A row whose followers' foreign
    key names a multi-table child, saved through its parent, resyncs
    none of them: only the parent's `post_save` fires, as in 10's case of a
    registered child. From the review, none changing
    what is sent: a bulk or cascade delete looks the followers up once per
    deleted row and schedules one callback per row, and a follower reached
    through several deleted rows of a chain is replaced once per row —
    batching per transaction belongs with 12b; each save now also walks
    the lookup paths of every registered model, which the index of the
    followed senders left open by 11a would avoid.
- [x] **11c. `depends_on` for custom extractors.** A custom extractor
  (`register_extractor`) has no `follow`: it declares what its documents
  read with `@rag.register_extractor(TextPlugin, depends_on=["page"])`,
  and a save or a delete of what it names resyncs it as 11a and 11b do.
  Requested by the demo project, whose receiver resyncing the text plugins
  of a saved page can now go. Decided in this feature:
  - **Three shapes of path**, from the registered model: a forward foreign
    key or one-to-one (`"page"`), a reverse foreign key or one-to-one on
    its own, by its accessor (`"text_plugins"` on `Page`), or a path
    through forward foreign keys only (`"product__category"`).
  - **Same machinery as 11a and 11b**: a forward path joins the lookup
    paths of 11b, a reverse relation the followed relations of 11a, with
    their batching, delete handling (`pre_delete`, before a `SET_NULL`),
    proxies and listeners connected only while needed. Followers that the
    model's `get_queryset()` filters out are sent with an empty group.
  - **Refused at registration**, with `ImproperlyConfigured`: `depends_on`
    that is not a list or a tuple, a link that is not a relation (an
    unknown name or a content field, at any depth), a many-to-many forward
    or reverse (left for 11d), a generic foreign key or `GenericRelation`,
    a path of several links that crosses a reverse relation (first link
    included), a path given twice, and `depends_on` while models are still
    loading. Every check runs before anything is registered.
- [x] **11c-bis. Resync through multi-column `ForeignObject` relations.**
  A `ForeignObject` over several columns (`Seminar.venue`, by
  `venue_city` and `venue_name`) is followed like a foreign key, in
  `follow`, in a lookup path (`"venue__name"`, or later in a path:
  `"seminar__venue__name"`) and in `depends_on`, forward or reverse.
  Registration already accepted it, but nothing resynced through it.
  Decided in this feature, after the review of 11c:
  - **Matched on every column**: a saved or deleted seminar resyncs the
    venue its columns name together, never another venue that shares
    one of them; a move resyncs both venues. A seminar whose columns name
    no venue resyncs nothing. A forward path already joined through the
    relation, so only the check that stopped at anything but a
    `ForeignKey` changed.
  - **Prefetch**: a reverse multi-column relation in `follow` is
    prefetched by all its target columns, and the children load every
    column they are matched by.
  - **`GenericRelation` still not followed**: it is a `ForeignObject`
    too, but one-to-many; only a forward many-to-one or one-to-one link
    counts as a foreign key, so a photo following its tags is not resynced
    and `Tag` keeps Django's fast delete.
  - **Left out**: deleting a venue that seminars follow is not tested
    (the test bench's relation cascades); a partly null set of columns
    is not tested; a `CompositePrimaryKey` is not covered. Renaming a
    venue (changing a column its seminars name it by) and the query count
    of the prefetch were left out too, then done in 11c-ter.
- [x] **11c-ter. Followers looked up before the save and at the commit.**
  Changing the columns that followers name a row by — a `to_field`, or the
  columns of a multi-column `ForeignObject` (renaming a venue) — resynced
  none of them: they were looked up after the save, by the new values. And
  forward followers were looked up at `post_save`, so rows attached later
  in the same transaction by a write that sends no signal (`bulk_create`,
  `QuerySet.update()`) were missed ([#26](https://github.com/gtolivier/django-model-rag/issues/26)).
  Decided in this feature, after the review of 11c-bis:
  - **Before the save**: `pre_save` looks up the followers reaching the row
    through foreign keys, by its primary key — the database still holds
    the old columns, so the row as committed is not loaded for them.
  - **At the commit**: a commit callback looks up the followers reaching
    the row through foreign keys then, and replaces their groups along with
    those of the followers found before the save and of those the row
    points to as saved — taken at `post_save`, so that a change left unsaved
    in memory afterwards is not followed. A row attached to it
    after the save by a write that sends no signal is resynced; so is one
    detached from it that way, through the list found before the save. A
    creation keeps the lookup at the save: nothing reaches the row through
    a foreign key yet.
  - **Same query count at the save**; the commit adds one lookup. A lookup
    failing at the commit is logged on the package logger with the saved
    row's key, does not escape the callback, and the other commit
    callbacks still run; the followers found before the save and at
    `post_save` are still replaced.
  - **Pinned**: syncing rows that follow a reverse multi-column relation
    runs as many queries for three children as for one — over two integer
    columns, with the new test-bench pair `Room` / `Booking`.
- [ ] **11d. Resync through many-to-many relations.** A many-to-many in
  `follow` or in a lookup path, forward or reverse, with `m2m_changed`
  (add, remove, clear) on top of the saves and deletes of both ends.
- [ ] **12a. Manual sync mode.** `MODEL_RAG_SYNC = "auto" | "notify" |
  "manual"` replaces `MODEL_RAG_SIGNALS`. In `manual`, nothing is connected
  — the `post_delete` listeners included, so every model keeps Django's fast
  delete, which `MODEL_RAG_SIGNALS = False` does not give back today. The
  wiring is decided at startup and redone on `setting_changed` (tests).
  Adds `rag.signals_paused()`, a context manager for a bulk import followed
  by a sync. A mode per model is not needed yet.
- [ ] **12b. Notify mode.** The package sends a signal of its own at the
  commit (`sources_changed`, say: the model and primary keys), once per
  transaction and grouped by model, with the deleted keys apart. In `auto`,
  the built-in receiver is connected to it and syncs; in `notify`, only
  detection and the signal run — no extraction, no output, no output
  check — and the project syncs when it wants (a task queue, a batch). Adds
  a public `SyncPipeline.run_pks(model, pks)`, which sends an empty group
  for a key whose row is gone. Settles the batching left open by 10 and
  11a.

## Not planned here

- **django-minimal-rag** gets its own roadmap after a design pass
  (embeddings, LLM, permission filtering, orphaned chunks). It has no
  prototype to rewrite.
- **Adapters for django CMS and Wagtail** will be separate packages, later.
