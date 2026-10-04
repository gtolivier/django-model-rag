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

documents = SyncPipeline().run()
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

## Where to register

Register your models in a `rag.py` module of your app, imported from the
app's `AppConfig.ready()`, when every model is loaded:

```python
# shop/apps.py
from django.apps import AppConfig


class ShopConfig(AppConfig):
    name = "shop"

    def ready(self) -> None:
        from . import rag  # noqa: F401 -- imported for its registrations
```

```python
# shop/rag.py
from django.contrib.auth import get_user_model

from django_model_rag import rag

from .models import Page, Product

rag.register(Product, follow=["category"])
rag.register(Page, follow=["text_plugins", "accordion_items"])
# A model from an app you do not own, registered from your own app.
rag.register(get_user_model(), fields=["first_name", "last_name"])
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

**Queries.** `SyncPipeline().run()` reads followed foreign keys and
one-to-one relations, and every relation along a lookup path, in the same
query as the instances (`select_related`), and each followed reverse foreign
key, many-to-many or generic relation in one more query
(`prefetch_related`), whatever the number of instances; instances are read
in chunks of 1000. `run_instance` makes at most one query per relation it crosses.

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
