# django-model-rag

Discover and normalize text content from any Django model, ready for a RAG
pipeline.

**It is** a registry and a set of extractors that turn instances of your
Django models into plain-text `NormalizedDocument`s — with a title, a
language, a stable source key and a resolved URL — so that a retrieval
pipeline can index them and cite real pages.

**It is not** a RAG pipeline. It does not chunk, embed, store vectors or call
an LLM. That is the job of a separate package,
[django-minimal-rag](https://github.com/gtolivier/django-minimal-rag), which
does not depend on this one.

## Status

Pre-alpha, not usable yet. The package is being written test-first, using
an earlier prototype as its behavioral specification. See the
[roadmap](ROADMAP.md) for what is done and the order of the features.

## Registering a model

```python
from django_model_rag import SyncPipeline, rag

rag.register(Product, fields=["name", "description"], title_field="name")
rag.register(Review, fields=["title", "product__name", "product__category__name"])
rag.register(Article)  # fields guessed
rag.register(Note, exclude=["internal_remarks"])

SyncPipeline(output).run()  # output: see "The output", below
```

Without `fields`, the text fields are guessed at registration, from their
type alone:

- **What is guessed:** a field whose type is exactly `CharField`, or a
  `TextField` or one of its subclasses. A `CharField` subclass is not
  guessed — `SlugField`, `EmailField`, `URLField`, and your own or a
  third-party package's subclasses too (translated fields, for instance).
  Name such a field in `fields=` to extract it. A primary key is never
  guessed: it identifies the instance, it is not its content.
- **In which order:** `title`, `name`, `heading` and `label` come first, in
  that order, then the other fields in declaration order. The first one
  gives the document title, unless `title_field` names another field.
- **Nothing is left out by name.** A text field holding a password hash, a
  token or private notes is guessed like any other. Leave it out with
  `exclude=`, or declare `fields=` instead.

`exclude` removes fields from the guessed ones and from nothing else: it
cannot add a field that is not guessed, and it cannot be combined with
`fields`. This differs from a `ModelForm`, whose `exclude` removes fields
from all its editable fields; both start from an implicit set, and here
that set is the guessed text fields. A model with nothing to guess, or whose
`exclude` leaves nothing, fails at registration with `ImproperlyConfigured`.

A model without `follow` or lookup paths can be registered in its own
`models.py`: guessing reads only that model's fields, not the app registry,
which Django is still loading then. Following relations and lookup paths
need every model loaded (see below).

## Fields of related objects

A name in `fields` — or `title_field` — can be a lookup path to one field of
a related object, as in `values()` or `list_display`:
`fields=["title", "product__name", "product__category__name"]`.

- **Links** go through foreign keys and one-to-one relations, forward or
  reverse, as many as needed. A reverse one-to-one is named by its query
  name, as in `values()` and `select_related` — its `related_query_name`, or
  its `related_name` when it has none, or else the model name in lower case.
  This differs from `follow`, which takes accessors. A foreign key may also
  be named by its column, as in `values()`: `category_id__name` reads like
  `category__name`.
- **The last name** is a content field of the model reached: its stripped
  text, or its label when it has choices.
- **Order and title.** A path keeps its place among the declared fields, and
  gives the title when it is declared first or named in `title_field`. A
  `title_field` path need not be in `fields`: it gives the title, not text.
- **Nothing to read adds nothing:** a null foreign key or a missing reverse
  one-to-one along the way.
- **With `follow`.** The two combine: a path picks one field of a related
  object, `follow` brings all its guessed text, and both can go through the
  same relation.

Errors are raised at registration, with `ImproperlyConfigured`, naming the
path: an unknown name along it, a link that is not a relation, a link that
holds several objects (a reverse foreign key, a many-to-many relation or a
generic relation: a path reads one value, use `follow` for those), a generic
foreign key, a path that ends on a relation, a path declared twice, a path
in `exclude` (which only names the model's own guessed fields), and a path
registered while models are still loading.

## Language, URL and permissions

```python
rag.register(Product, language_field="locale", url_field="permalink")
rag.register(Review, language_field="product__language")
rag.register(Article, language="fr", permissions=["blog.view_article"])
```

**Language.** Each document's `language` comes from, in this order:

- `language="fr"`: the same language, stripped, for every document of the
  model;
- `language_field`: the field it names, an own field or a lookup path;
- otherwise a guessed field: the model's own non-relation field named
  `language`, else `language_code`, else `lang` — chosen once per model, by
  name only;
- otherwise `None`.

The language is the stored value, stripped — a field with choices gives its
code, not its label. A configured or guessed field that is blank or null,
or a null foreign key along a path, gives `None`: there is no fallback to
the next source. The own field read as the language is left out of the
guessed text fields (name it in `fields=` to extract it too); with
`language="fr"`, a field named `language` stays a guessed text field.

**URL.** Each document's `url` is `url_field`'s value, stripped, when it is
given — an own field or a lookup path —, else the result of the model's
`get_absolute_url()`, else `""`. The URL is kept as stored or returned,
relative or absolute: nothing turns it into an absolute URL. A blank
`url_field`, or a null foreign key along its path, gives `""` without
falling back to `get_absolute_url()`; a `get_absolute_url()` that returns
`None` gives `""`, and one that raises lets its exception propagate out of
the pipeline. An own `url_field` is left out of the guessed text fields.

**Queries.** A `language_field` or `url_field` path is read in the same
query as the instances, like a path in `fields`.

**Permissions.** `permissions=["app_label.codename", ...]` puts the same
permission names, as a `frozenset`, on every document of the model, for the
retrieval side to filter on; without them, documents have an empty
`frozenset`. Nothing checks that the permissions exist: they are passed
through. A custom extractor gives each document its own with
`build_document(instance, text=..., permissions=[...])`; a bare string, or
an item that is not a string, raises `TypeError` there. A document's
`permissions` is typed as a set of strings (`collections.abc.Set[str]`), so
the retrieval side can compare it with set operators such as
`document.permissions <= user_permissions`.

Errors are raised at registration, with `ImproperlyConfigured`: a
`language_field` or `url_field` naming an unknown field — a method name
included — or a relation, a path refused for the same reasons as in
`fields`, `language` combined with `language_field`, a `language` that is
blank or not a string, `permissions` that is not a list or a tuple, and a
permission that is not a string of the form `app_label.codename`.

## Where to register

Add the package to `INSTALLED_APPS`, then register your models in a
`model_rag.py` module of your app. Once every model is loaded, the package
imports the `model_rag` module of each installed app that has one, as
`django.contrib.admin` does with `admin.py`. It does so in the order of
`INSTALLED_APPS`, and skips an app without one:

```python
# settings.py
INSTALLED_APPS = [
    # ...
    "django_model_rag",
    "shop",
]
```

```python
# shop/model_rag.py
from django.contrib.auth import get_user_model

from django_model_rag import rag

from .models import Page, Product

rag.register(Product, follow=["category"])
rag.register(Page, follow=["text_plugins", "accordion_items"])
# A model from an app you do not own, registered from your own app.
rag.register(get_user_model(), fields=["first_name", "last_name"])
```

**Changing a third-party app's registration.** Suppose an app registers its
models in its own `model_rag.py` and you want them indexed differently. Undo
its registration and register them again from an app listed after it in
`INSTALLED_APPS`, whose `model_rag.py` is imported later:

```python
# shop/model_rag.py — "shop" comes after "blog" in INSTALLED_APPS
from blog.models import Article

from django_model_rag import rag

rag.unregister(Article)
rag.register(Article, fields=["title", "body"])
```

## Following relations

`follow=[...]` appends the text of related objects to the text of the
model's own fields:

- **Names** are relation accessors, as in `prefetch_related`: a foreign key,
  a one-to-one or a many-to-many field by its name, a reverse relation by
  its `related_name`, or by its default accessor `<model>_set` when it has
  none (`remark_set`, not the query name `remark`). A `GenericRelation` is
  followed by its name, like a reverse foreign key; a `GenericForeignKey`
  cannot be followed, since its related model changes from one row to the
  next (write a custom extractor for it).
- **What a related object brings:** its guessed text fields, by the rules
  above, title-like ones first. `fields` and `exclude` apply to the model's
  own fields only, never to the related objects'.
- **In which order:** the model's own fields, then each relation in the
  order of `follow`, its objects in primary key order; every piece is
  separated by a blank line.
- **Nothing to follow adds nothing:** a null foreign key, a missing reverse
  one-to-one, an empty relation, or related objects whose text fields are
  blank.
- **One level only.** Only relations of the model itself are followed; to
  read further, name a field of a related object in `fields` (see above).
- **Title.** A model with no text field of its own — or whose `exclude`
  leaves none — can be registered when it follows relations; its document
  title is `str(instance)`. A model whose own fields are blank but whose
  related text is not still gets a document, with a blank title.

Errors are raised at registration, with `ImproperlyConfigured`: `follow`
that is not a list or a tuple, a name that is not a relation accessor of the
model, a relation followed twice, a generic foreign key, a related model with
no text field, and `follow` while models are still loading.

**A related object brings all its guessed text fields, sensitive ones
included.** `follow=["author"]` towards Django's `User` brings its
`password` field — the hash is a plain `CharField` — along with the name and
the e-mail address. `exclude` cannot leave it out, since it only applies to
the model's own fields. Follow only relations to models whose whole text you
want indexed; for the others, pick their fields with lookup paths
(`fields=["title", "author__first_name", "author__last_name"]`), or write a
custom extractor.

**Queries.** `run()` reads followed foreign keys and
one-to-one relations, and every relation along a lookup path, in the same
query as the instances (`select_related`), and each followed reverse foreign
key, many-to-many or generic relation in one more query
(`prefetch_related`), whatever the number of instances; instances are read
in chunks of 1000. `run_instance` makes one query to ask the extractor's
queryset whether it keeps the instance, then at most one query per relation
it crosses.

Those queries load only the columns the documents read: the declared
fields, the title, language and URL fields, the related columns a lookup
path names, and the text fields of a followed relation with what links it
back — including a column other than the primary key that the link targets
(`to_field`). A relation the default manager already joins with
`select_related`, the model's or a followed related model's, stays loaded
too. A model whose `get_absolute_url()` builds the URL (no `url_field`), or
whose title falls back on `str(instance)`, keeps all its own columns, since
either may read any of them. A custom extractor shapes the queryset its
instances are loaded from by overriding `get_queryset(queryset)` — to add
`select_related` or `prefetch_related`, say — and must return a `QuerySet`
of the model's instances: a `values()` queryset, or another model's, raises
`TypeError`. The documents stay in primary key order whatever order the hook
sets, and an instance repeated by a join is extracted once. An instance
the hook filters out is not extracted: `run()` skips it and prunes its
documents, and `run_instance`, which already holds the instance, only asks
the hook's queryset whether it keeps it, and sends an empty group if not.

## The output

`SyncPipeline(output)` hands the documents to an output that your project
supplies — [django-minimal-rag](https://github.com/gtolivier/django-minimal-rag)
is one. Any object with these two methods is an output; the
`DocumentOutput` Protocol, importable from `django_model_rag`, lets a type
checker verify it:

```python
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from django_model_rag import NormalizedDocument


class MyOutput:
    def replace(self, groups: Mapping[str, Sequence[NormalizedDocument]]) -> None:
        """Replace the stored documents of each source key with its group."""

    def prune(self, model_label: str, kept_keys: AbstractSet[str]) -> None:
        """Delete the documents of model_label whose source key is not kept."""
```

- **By source.** A group is the complete set of documents of one instance,
  keyed by its `source_key`: it replaces everything the output holds for
  that instance. An empty group removes it.
- **`run()`** sends each model's groups in batches, one `replace()` per
  chunk of 1000 instances, then calls `prune()` with the model's label
  (`app_label.model_name`) and the keys of the instances that produced
  documents, so that an instance deleted, filtered out or now producing
  nothing is removed. If an extractor raises, the exception propagates:
  the batches already sent stay sent, and that model is not pruned.
  Running again is the retry.
- **`run_instance(instance)`** sends that instance's group, even empty, so
  that an instance whose extractor now returns `None`, or that its
  extractor's queryset filters out, is removed. It never prunes, and sends
  nothing if the extractor raises. An unsaved instance (no primary key)
  raises `ValueError`.
- **A document's source is the instance it was extracted from.** An
  extractor returning a document whose source is another instance fails
  with `TypeError`: it would replace that other instance's documents.

`run()` and `run_instance()` return `None`: the documents go only to the
output.

## Synchronizing: `sync_model_rag`

```sh
python manage.py sync_model_rag                    # every registered model
python manage.py sync_model_rag shop.product blog.article
```

The command runs the pipeline (`run()`, with its final `prune()`) into the
output named by the `MODEL_RAG_OUTPUT` setting. It runs every registered
model, in registration order, or only the models it is given, as
`app_label.model_name` and in the given order; a model named twice runs once,
at its first position. With no model registered, it writes a warning on
stderr and ends without error. Run it once after installing
the package or registering a new model: until then, the output holds none
of that model's documents. Run it again whenever you want to repair what
saving instances did not keep up to date.

The setting has the shape of Django's `STORAGES`:

```python
MODEL_RAG_OUTPUT = {
    "BACKEND": "myproject.rag.MyOutput",  # a dotted path to the output class
    "OPTIONS": {"collection": "site"},  # optional: keyword arguments
}
```

The command imports `BACKEND`, builds a new instance with `OPTIONS` as its
keyword arguments, and checks the class has a callable `replace` and
`prune`. It does all this before running any model. A missing setting,
a setting that is not a dict or has no `BACKEND`, a `BACKEND` that is not a
string, cannot be imported or names something other than a class, `OPTIONS`
that are not a dict, and a class without one of the two methods fail with
`ImproperlyConfigured`. So does a label naming no model of an installed app,
or a model that is not registered, but with `CommandError`. Both are raised
before anything is sent.

**Trying it out.** `django_model_rag.output.ConsoleOutput` writes what it
receives to standard output, or to the stream it is given
(`ConsoleOutput(stream=...)`): each source key, then its documents' titles and
texts, `<key> removed` for an empty group, and `<model label> kept <n>` for
each prune:

```python
MODEL_RAG_OUTPUT = {"BACKEND": "django_model_rag.output.ConsoleOutput"}
```

**Failures.** When a model's run raises, the command writes
`<app_label.model_name>: <ExceptionType>: <message>` on stderr and goes on
with the next model. That model is not pruned, and the batches it already
sent stay sent (see "The output", above). Each model that succeeds writes
`<app_label.model_name>: synced` on stdout, except with `--verbosity 0`,
which still writes failures. With `--traceback`, each failure line is
followed by its traceback. If any model failed, the command
ends with a `CommandError` naming every failed model, in run order, so it
exits with a non-zero status. Running it again, or with only the failed
models, is the retry.

## Keeping in step: the signals

Once the package is in `INSTALLED_APPS`, saving or deleting an instance of a
registered model updates the output named by `MODEL_RAG_OUTPUT`:

- **A save** replaces the instance's group with its documents, even an empty
  group when the extractor now returns nothing.
- **A delete** — of an instance or of a queryset — replaces each deleted
  instance's group with an empty one, sent straight to the output: the
  extractor and its `get_queryset` are not called, as the row is gone.

**After the commit.** Nothing is sent while the transaction is open: the
signal schedules the work with `transaction.on_commit`, and the instance is
reloaded and extracted then. The documents are those of the committed
instance, including what the same transaction changed after the save —
the inlines and many-to-many relations an admin form saves, for instance. A
rolled-back transaction sends nothing; outside a transaction (autocommit),
the work runs at once. Each commit builds a new output from
`MODEL_RAG_OUTPUT`, with its `OPTIONS`, and the work runs in the process
that commits: a slow output delays the response that saved the instance.
The signals follow the default database only, for now: saves and deletes on
another database alias are not supported yet.

**Proxies and multi-table inheritance.** Saving or deleting through a proxy
of a registered model updates the registered model's group, under its own
label. Saving a multi-table child updates each registered model among the
child and its parents, each under its own label; deleting it empties the
group of the nearest registered one. A proxy defined after its concrete
model is registered gets no delete listener, so deleting through it empties
nothing: register from `model_rag.py`, once every model is loaded. A proxy
registered instead of its concrete model is not updated by the signals.

**What is not sent.** Unregistered models: they keep Django's fast delete,
since the package listens to `post_delete` only for registered models and
their proxies. Raw saves, such as `loaddata` loading a fixture. Changes the
ORM signals do not see: `QuerySet.update()`, `bulk_create()`, raw SQL. A
change to a related object whose text a registered model reads — through
`follow` or a lookup path — leaves that model's documents stale until they
are saved again. For all of these, run `sync_model_rag`, or call
`SyncPipeline(output).run_instance(instance)` from a receiver of your own.

**Failures.** An extractor, an output or a database error reloading the
instance that raises at the commit is logged with `logger.exception` on the
`django_model_rag` logger, naming the instance's source key, and the commit
goes on: the instance keeps its previous documents until its next save or
the next `sync_model_rag`, and the other instances of the transaction are
still sent. A missing or invalid `MODEL_RAG_OUTPUT` — including `OPTIONS`
its `BACKEND` class does not accept — is not logged: it raises
`ImproperlyConfigured` at the save or the delete, so that a forgotten
setting cannot silently stop the indexing. A save checks it in `pre_save`,
before the row is written, so that it writes nothing even in autocommit; a
delete raising this way is rolled back.

**Turning them off.** `MODEL_RAG_SIGNALS = False` (default `True`) makes the
signals send nothing and check nothing. Your test settings need either
that, or a `MODEL_RAG_OUTPUT` pointing to a test output — as with
`EMAIL_BACKEND` — or every save of a registered model in your tests raises
`ImproperlyConfigured`.

```python
# settings_test.py
MODEL_RAG_SIGNALS = False
```

## Requirements

- Python 3.11+
- Django 5.2 LTS, 6.0 or 6.1

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
```

## License

MIT — see [LICENSE](LICENSE).
