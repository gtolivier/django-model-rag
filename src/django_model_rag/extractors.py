"""The extractors: the base class of custom ones, and the one of declared fields."""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeGuard, TypeVar, cast

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import (
    CharField,
    Field,
    ForeignKey,
    ManyToManyField,
    Model,
    Prefetch,
    QuerySet,
    TextField,
)
from django.db.models.constants import LOOKUP_SEP
from django.db.models.fields.reverse_related import ForeignObjectRel
from django.db.models.manager import BaseManager

from django_model_rag.documents import NormalizedDocument

M = TypeVar("M", bound=Model)

_TITLE_LIKE_NAMES = ("title", "name", "heading", "label")
"""The names of the guessed fields that come first, in this order."""


def _title_rank(name: str) -> int:
    """Rank ``name``: title-like names first, in order, then the others."""
    if name in _TITLE_LIKE_NAMES:
        return _TITLE_LIKE_NAMES.index(name)
    return len(_TITLE_LIKE_NAMES)


# quoted: Django's Field is generic for the type checker only, and before
# Python 3.14 an annotation is evaluated when the function is defined
def _is_text_field(field: "Field[Any, Any]") -> bool:
    """Tell whether ``field`` holds text content."""
    # a primary key is an identifier, not content
    if field.primary_key:
        return False
    # a CharField subclass is a kind of field of its own, such as a code,
    # an identifier, an address or a link, not content
    return type(field) is CharField or isinstance(field, TextField)


def text_fields(model: type[Model]) -> list[str]:
    """List the names of ``model``'s text fields, title-like names first.

    The others follow in declaration order.
    """
    names = [
        field.name
        # unlike get_fields(), concrete_fields needs no loaded app registry,
        # so a models.py can register its models while Django loads the apps;
        # it is not in the documented meta API, but Django itself relies on it
        for field in model._meta.concrete_fields
        if _is_text_field(field)
    ]
    return sorted(names, key=_title_rank)  # stable sort


class BaseExtractor(ABC, Generic[M]):
    """Build the document(s) of an instance of a model."""

    @abstractmethod
    def extract(
        self, instance: M
    ) -> NormalizedDocument | Iterable[NormalizedDocument] | None:
        """Build the document(s) of ``instance``, or nothing to skip it."""

    def get_queryset(self, queryset: QuerySet[M]) -> QuerySet[M]:
        """Shape how ``queryset`` loads the instances to extract, not which it holds.

        A run of a single instance, already loaded, does not go through it.
        """
        return queryset

    def build_document(  # noqa: PLR0913  # one keyword per document field
        self,
        instance: M,
        *,
        text: str,
        title: str = "",
        url: str = "",
        language: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        permissions: Collection[str] = (),
    ) -> NormalizedDocument:
        """Build a document with ``text``, its source taken from ``instance``."""
        return NormalizedDocument(
            text=text,
            source_app_label=instance._meta.app_label,  # Django's public meta API
            source_model=instance._meta.model_name or "",  # Django's public meta API
            source_pk=instance.pk,
            title=title,
            url=url,
            language=language,
            metadata=metadata or {},
            # The document accepts any collection and stores a frozenset.
            permissions=cast("AbstractSet[str]", permissions),
        )


_FIELD_SEPARATOR = "\n\n"


def accessor_name(relation: "Field[Any, Any] | ForeignObjectRel") -> str | None:
    """Name the attribute that crosses ``relation`` on an instance, if it has one.

    A reverse relation is reached by its accessor, which ``related_name`` may
    set apart from its query name; a forward one by its name.
    """
    if isinstance(relation, ForeignObjectRel):
        return relation.get_accessor_name()
    return relation.name


def query_name(relation: "Field[Any, Any] | ForeignObjectRel") -> str:
    """Name ``relation`` as select_related() does."""
    # select_related() names a reverse relation by its query name, which
    # related_query_name may set apart from its accessor
    return relation.name


def relations_by_accessor(
    model: type[Model],
) -> "dict[str, Field[Any, Any] | ForeignObjectRel]":
    """Map each of ``model``'s relations, forward or reverse, by its accessor.

    A forward relation is its field, under its name; a reverse one is its
    relation object, under the accessor ``related_name`` may set.
    """
    relations: dict[str, Field[Any, Any] | ForeignObjectRel] = {}
    for field in model._meta.get_fields():
        if field.is_relation and (accessor := accessor_name(field)) is not None:
            relations[accessor] = field
    return relations


@dataclass(frozen=True)
class PathLink:
    """A relation a lookup path crosses, as the path names it."""

    model: type[Model]
    """The model the link starts from."""
    name: str
    """The link as written in the path: a query name, or a column name."""
    # quoted: Django's Field is generic for the type checker only
    relation: "Field[Any, Any] | ForeignObjectRel"
    """The relation the link crosses, forward or reverse."""

    @property
    def accessor(self) -> str:
        """Name the attribute that crosses the link on an instance."""
        # a foreign key named by its column, such as category_id, is read as
        # the related object, as in values()
        return accessor_name(self.relation) or self.name

    @property
    def query_name(self) -> str:
        """Name the link as select_related() does."""
        return query_name(self.relation)


def path_links(model: type[Model], path: str) -> Iterator[PathLink]:
    """Walk the links of the lookup ``path`` from ``model``, up to its last field.

    The walk ends after a link that leads to no single model: a field that is
    not a relation, or a generic foreign key.

    Raises:
        FieldDoesNotExist: a link is not a field of the model it starts from.
    """
    *names, _ = path.split(LOOKUP_SEP)
    for name in names:
        relation = model._meta.get_field(name)
        yield PathLink(model, name, relation)
        related_model = relation.related_model
        if related_model is None:
            return
        model = related_model


def _related_instance(instance: Model, accessor: str) -> Model | None:
    """Cross the relation ``accessor`` of ``instance``, or ``None`` if unset.

    A missing reverse one-to-one raises instead of returning ``None``.
    """
    try:
        related: Model | None = getattr(instance, accessor)
    except ObjectDoesNotExist:
        return None
    return related


def _field_text(instance: Model, name: str) -> str:
    """Read the own field ``name`` of ``instance`` as stripped text.

    A field with choices reads as its label.
    """
    value = getattr(instance, name)
    # Django adds get_<name>_display only to fields that have choices
    if getattr(instance._meta.get_field(name), "choices", None):
        value = getattr(instance, f"get_{name}_display")()
    return _stripped_text(value)


def _stripped_text(value: object) -> str:
    """Read a stored ``value`` as stripped text, empty when it is unset."""
    return "" if value is None else str(value).strip()


def _document_text(field_texts: list[str]) -> str:
    """Join the non-empty ``field_texts``, in order."""
    return _FIELD_SEPARATOR.join(text for text in field_texts if text)


_GUESSED_LANGUAGE_FIELDS = ("language", "language_code", "lang")
"""The names of an own field read as the language when none is declared.

The first name the model has wins.
"""


def guessed_language_field(model: type[Model]) -> str | None:
    """Name the own field of ``model`` guessed, by its name, as the language."""
    # concrete_fields, not get_fields(): it needs no loaded app registry
    own_fields = {
        field.name for field in model._meta.concrete_fields if not field.is_relation
    }
    return next((name for name in _GUESSED_LANGUAGE_FIELDS if name in own_fields), None)


# a select_related() lookup, mapped to the related model and its fields read
_ReadByPrefix: TypeAlias = dict[str, tuple[type[Model], set[str]]]


def _lookup_paths(read_fields: Iterable[str]) -> list[str]:
    """Return, once each, the lookup paths among the ``read_fields``."""
    return [name for name in dict.fromkeys(read_fields) if LOOKUP_SEP in name]


def _is_selected(relation: "Field[Any, Any] | ForeignObjectRel") -> bool:
    """Tell whether ``relation`` leads to a single object, read in the same query.

    These are the foreign keys and the one-to-one relations, forward or reverse.
    """
    if isinstance(relation, ForeignObjectRel):
        return bool(relation.one_to_one)
    # a generic foreign key leads to a single object too, but has no column
    # select_related() could join on
    return bool(relation.concrete and (relation.many_to_one or relation.one_to_one))


def _is_prefetched(relation: "Field[Any, Any] | ForeignObjectRel") -> bool:
    """Tell whether ``relation`` leads to many objects, read in one more query.

    These are the reverse foreign keys, the many-to-many relations, forward or
    reverse, and the generic relations.
    """
    return bool(relation.one_to_many or relation.many_to_many)


def _is_reverse_foreign_key(
    relation: "Field[Any, Any] | ForeignObjectRel | None",
) -> TypeGuard[ForeignObjectRel]:
    """Tell whether ``relation`` is a reverse foreign key, to the related objects."""
    return isinstance(relation, ForeignObjectRel) and bool(relation.one_to_many)


def _sorted_relations(
    model: type[Model], followed: Sequence[str]
) -> "tuple[list[str], list[str | Prefetch[Any]]]":
    """Sort the relations ``followed`` from ``model`` by how they are read.

    Returns:
        The names to give select_related(), then the lookups to give
        prefetch_related().
    """
    relations = relations_by_accessor(model)
    selected: list[str] = []
    prefetched: list[str | Prefetch[Any]] = []
    for accessor in followed:
        relation = relations.get(accessor)
        if relation is None:
            continue
        if _is_selected(relation):
            selected.append(query_name(relation))
        elif _is_prefetched(relation):
            prefetched.append(_prefetch(accessor, relation))
    return selected, prefetched


def _link_back_fields(
    relation: "Field[Any, Any] | ForeignObjectRel",
) -> list[str] | None:
    """Return the fields of the related model that link it back through ``relation``.

    These are the foreign key of a reverse foreign key, the object_id and
    content_type of a generic relation, and none for a many-to-many; None for
    any other relation.
    """
    # imported here: contenttypes' models cannot load before the apps are ready,
    # and this module is imported from models modules
    from django.contrib.contenttypes.fields import GenericRelation  # noqa: PLC0415

    if _is_reverse_foreign_key(relation):
        return [relation.field.name]
    if isinstance(relation, GenericRelation):
        return [relation.object_id_field_name, relation.content_type_field_name]
    if relation.many_to_many:
        # the join table links it back, not a column of the related model
        return []
    return None


def _prefetch(
    accessor: str, relation: "Field[Any, Any] | ForeignObjectRel"
) -> "str | Prefetch[Any]":
    """Return what to give prefetch_related() for ``relation``, as ``accessor``.

    A reverse foreign key, a generic relation or a many-to-many loads only its
    text columns and the fields that link it back.
    """
    link_back = _link_back_fields(relation)
    related = relation.related_model
    # for the type checker only: a relation that links back leads to a model
    if link_back is None or not isinstance(related, type):
        return accessor
    manager_queryset = related._default_manager.all()
    # a foreign key the manager already joins cannot be deferred
    joined = _joined_relations(manager_queryset)
    return Prefetch(
        accessor,
        queryset=manager_queryset.only(
            "pk", *text_fields(related), *link_back, *joined
        ),
    )


def _selected_run(model: type[Model], path: str) -> list[PathLink]:
    """Return the single-object relations ``path`` starts with."""
    run: list[PathLink] = []
    for link in path_links(model, path):
        if not _is_selected(link.relation):
            break
        run.append(link)
    return run


def _query_path(run: Sequence[PathLink]) -> str:
    """Join the links of ``run`` into the lookup select_related() names it by."""
    return LOOKUP_SEP.join(link.query_name for link in run)


def _selected_path_prefixes(model: type[Model], paths: Iterable[str]) -> list[str]:
    """Return the run of single-object relations each lookup path starts with."""
    runs = (_selected_run(model, path) for path in paths)
    return [_query_path(run) for run in runs if run]


def _followed_read(model: type[Model], followed: Sequence[str]) -> _ReadByPrefix:
    """Map each ``followed`` selected relation to its model and its text fields."""
    relations = relations_by_accessor(model)
    read: _ReadByPrefix = {}
    for accessor in followed:
        relation = relations.get(accessor)
        if relation is None or not _is_selected(relation):
            continue
        owner = relation.related_model
        # for the type checker only: a selected relation leads to a model
        if owner is not None:
            read[query_name(relation)] = (owner, set(text_fields(owner)))
    return read


def _read_by_prefix(
    model: type[Model], paths: Iterable[str], followed: Sequence[str]
) -> _ReadByPrefix:
    """Map each selected relation followed or prefixing a lookup path to its reads.

    The value is the related model, and the names of its fields read: its text
    fields for a followed relation, those the paths name for a prefix.
    """
    read = _followed_read(model, followed)
    for path in paths:
        run = _selected_run(model, path)
        names = path.split(LOOKUP_SEP)
        for depth, link in enumerate(run, start=1):
            owner = link.relation.related_model
            # for the type checker only: a selected relation leads to a model
            if owner is None:
                break
            prefix = _query_path(run[:depth])
            read.setdefault(prefix, (owner, set()))[1].add(names[depth])
    return read


def _unread_related_columns(
    model: type[Model], paths: Iterable[str], followed: Sequence[str]
) -> list[str]:
    """Return the lookup names of the selected related columns never read."""
    return [
        f"{prefix}{LOOKUP_SEP}{name}"
        for prefix, (owner, read) in _read_by_prefix(model, paths, followed).items()
        for name in _unread_field_names(owner, read)
    ]


def _unread_field_names(model: type[Model], read: set[str]) -> list[str]:
    """Return the names of ``model``'s columns that ``read`` does not name.

    The primary key is always read. A foreign key may be named by its field
    or by its column.
    """
    return [
        field.name
        for field in model._meta.concrete_fields
        if not field.primary_key and not {field.name, field.attname} & read
    ]


def _unread_columns(
    model: type[Model], read_fields: Iterable[str], followed: Sequence[str]
) -> list[str]:
    """Return the names of ``model``'s own columns never read.

    The first link of each lookup path among the ``read_fields``, and each
    relation ``followed``, are read: they stay, with the own fields named and
    the columns the followed relations' prefetches match by.
    """
    read = {name.split(LOOKUP_SEP)[0] for name in (*read_fields, *followed)}
    read |= _prefetch_match_columns(model, followed)
    return _unread_field_names(model, read)


def _joined_relations(queryset: QuerySet[Model]) -> list[str]:
    """Return the relations ``queryset`` already select_related() by name.

    A select_related() without a field names none.
    """
    joined = queryset.query.select_related
    return list(joined) if isinstance(joined, dict) else []


def _joined_paths(joined: object, prefix: str = "") -> set[str]:
    """Return every lookup path a select_related() mapping joins, nested too."""
    if not isinstance(joined, dict):
        return set()
    paths: set[str] = set()
    for name, deeper in joined.items():
        path = f"{prefix}{name}"
        paths.add(path)
        paths |= _joined_paths(deeper, f"{path}{LOOKUP_SEP}")
    return paths


def _deferrable_related_columns(
    queryset: QuerySet[Model], paths: Iterable[str], followed: Sequence[str]
) -> list[str]:
    """Return the selected related columns never read that ``queryset`` can defer.

    A relation the queryset joins deeper cannot be deferred.
    """
    joined = _joined_paths(queryset.query.select_related)
    unread = _unread_related_columns(queryset.model, paths, followed)
    return [name for name in unread if name not in joined]


def _prefetch_match_columns(model: type[Model], followed: Sequence[str]) -> set[str]:
    """Return the names of the columns the ``followed`` prefetches match by."""
    relations = relations_by_accessor(model)
    return {
        column
        for accessor in followed
        if (column := _prefetch_match_column(relations.get(accessor))) is not None
    }


def _prefetch_match_column(
    relation: "Field[Any, Any] | ForeignObjectRel | None",
) -> str | None:
    """Return the name of the parent's column a prefetch of ``relation`` matches by.

    None when ``relation`` is not a reverse foreign key or a many-to-many.
    """
    # the prefetch matches the related objects to their parent by the column
    # the foreign key to the parent targets, which may not be the primary key:
    # the reverse foreign key itself, or the through model's for a many-to-many
    if _is_reverse_foreign_key(relation):
        return relation.field.target_field.name
    if not isinstance(relation, ManyToManyField):
        return None
    through = relation.remote_field.through
    key = through._meta.get_field(relation.m2m_field_name()) if through else None
    return key.target_field.name if isinstance(key, ForeignKey) else None


class DeclaredFieldsExtractor(BaseExtractor[Model]):
    """Build one document from the fields a model declares when registered."""

    def __init__(  # noqa: PLR0913, PLR0917  # one argument per registration option
        self,
        fields: tuple[str, ...],
        title_field: str | None,
        follow: tuple[str, ...] = (),
        language_field: str | None = None,
        language: str | None = None,
        url_field: str | None = None,
        permissions: tuple[str, ...] = (),
    ) -> None:
        """Read ``fields`` as the text, ``title_field`` (if any) as the title.

        The text of the relations in ``follow`` comes after the fields.
        ``language_field`` (if any) names the field holding the language,
        declared or guessed: an own field, or a lookup path such as
        ``page__language``.
        ``language`` (if any), stripped, is the constant language of every
        document.
        ``url_field`` (if any) names the field whose stripped value is the
        document url: an own field, or a lookup path such as
        ``bookmark__link``.
        ``permissions`` are given to every document.
        """
        self.fields = fields
        self.title_field = title_field
        self.follow = follow
        self.language_field = language_field
        self.language = None if language is None else language.strip()
        self.url_field = url_field
        self.permissions = permissions
        # listed once per related model, and resolved once per lookup path;
        # the registry builds a fresh extractor for each run, so a redefined
        # model is never served stale
        self._related_text_fields: dict[type[Model], list[str]] = {}
        self._path_accessors: dict[tuple[type[Model], str], list[str]] = {}

    def extract(self, instance: Model) -> NormalizedDocument | None:
        """Build the document of ``instance``, or nothing when it has no text.

        Its text is that of its fields, then of its followed relations.
        """
        field_texts = [self._declared_text(instance, name) for name in self.fields]
        followed_texts = [self._followed_text(instance, name) for name in self.follow]
        text = _document_text(field_texts + followed_texts)
        if not text:
            return None
        return self.build_document(
            instance,
            text=text,
            title=self._title(instance, field_texts),
            url=self._url(instance),
            language=self._language(instance),
            permissions=self.permissions,
        )

    def get_queryset(self, queryset: QuerySet[Model]) -> QuerySet[Model]:
        """Load only the columns and relations its fields and relations read.

        The followed foreign keys and one-to-one relations, and the run of them
        each lookup path starts with, come with each instance, in the same query.
        The followed reverse foreign keys, many-to-many relations, forward or
        reverse, and generic relations come in one more query each for all the
        instances.
        """
        model = queryset.model
        read_fields = [*self.fields, *self.single_fields]
        if self._reads_only_named_columns(model):
            # a foreign key the queryset already joins cannot be deferred
            kept = [*read_fields, *_joined_relations(queryset)]
            queryset = queryset.defer(*_unread_columns(model, kept, self.follow))
        selected, prefetched = _sorted_relations(model, self.follow)
        paths = _lookup_paths(read_fields)
        selected.extend(_selected_path_prefixes(model, paths))
        # never select_related() without a field: it would follow every non-null
        # foreign key, followed or not
        if selected:
            queryset = queryset.select_related(*selected)
            if unread := _deferrable_related_columns(queryset, paths, self.follow):
                queryset = queryset.defer(*unread)
        if prefetched:
            queryset = queryset.prefetch_related(*prefetched)
        return queryset

    @property
    def single_fields(self) -> list[str]:
        """List the fields its title, language and url options name, if set."""
        named = (self.title_field, self.language_field, self.url_field)
        return [name for name in named if name is not None]

    def _reads_only_named_columns(self, model: type[Model]) -> bool:
        """Tell whether the own columns of ``model`` it reads are all named by it.

        Only the fields it declares name them, and get_absolute_url() may read any
        column, when no URL field replaces it.
        """
        return bool(self.fields) and bool(
            self.url_field or not hasattr(model, "get_absolute_url")
        )

    def _url(self, instance: Model) -> str:
        """Give the url of ``instance``: its url field, else ``get_absolute_url``."""
        if self.url_field:
            return self._stored_text(instance, self.url_field)
        get_absolute_url = getattr(instance, "get_absolute_url", None)
        if get_absolute_url is None:
            return ""
        url = get_absolute_url()
        return "" if url is None else str(url)

    def _language(self, instance: Model) -> str | None:
        """Give the constant language, else read ``instance``'s language field.

        That field is the declared one, or the one the registry guessed by name.
        """
        if self.language is not None:
            return self.language
        if self.language_field is None:
            return None
        return self._stored_text(instance, self.language_field) or None

    def _declared_text(self, instance: Model, path: str) -> str:
        """Read the declared field ``path`` of ``instance`` as stripped text.

        ``path`` may be a lookup path, such as ``category__name``, to read a
        field of a related instance.
        """
        path_end = self._path_end(instance, path)
        if path_end is None:
            return ""
        owner, name = path_end
        return _field_text(owner, name)

    def _stored_text(self, instance: Model, path: str) -> str:
        """Read the stored value at ``path`` from ``instance`` as stripped text.

        Unlike ``_declared_text``, a field with choices reads as its stored
        value, not its label. Empty when a link of ``path`` is unset.
        """
        path_end = self._path_end(instance, path)
        if path_end is None:
            return ""
        owner, name = path_end
        return _stripped_text(getattr(owner, name))

    def _path_end(self, instance: Model, path: str) -> tuple[Model, str] | None:
        """Give the instance holding the last field of ``path``, and its name.

        ``None`` when a link of ``path`` from ``instance`` is unset.
        """
        *_, name = path.split(LOOKUP_SEP)
        for accessor in self._accessors(type(instance), path):
            related = _related_instance(instance, accessor)
            if related is None:
                return None
            instance = related
        return instance, name

    def _accessors(self, model: type[Model], path: str) -> list[str]:
        """List the attributes crossing the links of ``path`` from ``model``.

        They are resolved on first use.
        """
        key = (model, path)
        if key not in self._path_accessors:
            self._path_accessors[key] = [
                link.accessor for link in path_links(model, path)
            ]
        return self._path_accessors[key]

    def _followed_text(self, instance: Model, name: str) -> str:
        """Read the text of the relation ``name`` of ``instance``.

        A reverse one-to-one without an object raises, where it has no text.
        """
        try:
            related = getattr(instance, name)
        except ObjectDoesNotExist:
            return ""
        return self._related_text(related)

    def _related_text(self, related: Model | BaseManager[Model] | None) -> str:
        """Join the texts of the text fields of ``related``, title-like ones first.

        ``related`` may be the manager of a reverse relation: its objects follow
        one another, in primary key order.
        """
        if related is None:
            return ""
        if isinstance(related, BaseManager):
            # sorted in Python, not order_by(): a prefetched relation stays prefetched
            items = sorted(related.all(), key=lambda item: item.pk)
            return _document_text([self._related_text(item) for item in items])
        return _document_text(
            [_field_text(related, name) for name in self._text_fields_of(type(related))]
        )

    def _text_fields_of(self, model: type[Model]) -> list[str]:
        """List the text fields of the related ``model``, guessed on first use."""
        if model not in self._related_text_fields:
            self._related_text_fields[model] = text_fields(model)
        return self._related_text_fields[model]

    def _title(self, instance: Model, field_texts: list[str]) -> str:
        """Take the text of the title field, or the first of ``field_texts``.

        Without either, ``str(instance)`` gives the title: a model that only
        follows relations has no own text to take it from.

        ``field_texts`` are the texts of the declared fields, in order: a title
        field among them is not read a second time.
        """
        if self.title_field is None:
            return field_texts[0] if field_texts else str(instance)
        if self.title_field in self.fields:
            return field_texts[self.fields.index(self.title_field)]
        return self._declared_text(instance, self.title_field)
