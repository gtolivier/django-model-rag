"""The extractors: the base class of custom ones, and the one of declared fields."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import CharField, Field, Model, TextField
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

    def build_document(  # noqa: PLR0913  # one keyword per document field
        self,
        instance: M,
        *,
        text: str,
        title: str = "",
        url: str = "",
        language: str | None = None,
        metadata: Mapping[str, Any] | None = None,
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
        # select_related() names a reverse relation by its query name, which
        # related_query_name may set apart from its accessor
        return self.relation.name


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


class DeclaredFieldsExtractor(BaseExtractor[Model]):
    """Build one document from the fields a model declares when registered."""

    def __init__(
        self,
        fields: tuple[str, ...],
        title_field: str | None,
        follow: tuple[str, ...] = (),
        language_field: str | None = None,
    ) -> None:
        """Read ``fields`` as the text, ``title_field`` (if any) as the title.

        The text of the relations in ``follow`` comes after the fields.
        ``language_field`` (if any) names the own field holding the language.
        """
        self.fields = fields
        self.title_field = title_field
        self.follow = follow
        self.language_field = language_field
        # guessed once per related model, and resolved once per lookup path;
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
            language=self._language(instance),
        )

    def _language(self, instance: Model) -> str | None:
        """Read the language field of ``instance``, declared or guessed by name."""
        own_fields = {
            field.name for field in instance._meta.get_fields() if not field.is_relation
        }
        candidates = (
            (self.language_field,) if self.language_field else _GUESSED_LANGUAGE_FIELDS
        )
        name = next(
            (candidate for candidate in candidates if candidate in own_fields), None
        )
        if name is None:
            return None
        return _stripped_text(getattr(instance, name)) or None

    def _declared_text(self, instance: Model, path: str) -> str:
        """Read the declared field ``path`` of ``instance`` as stripped text.

        ``path`` may be a lookup path, such as ``category__name``, to read a
        field of a related instance.
        """
        *_, name = path.split(LOOKUP_SEP)
        for accessor in self._accessors(type(instance), path):
            related = _related_instance(instance, accessor)
            if related is None:
                return ""
            instance = related
        return _field_text(instance, name)

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
