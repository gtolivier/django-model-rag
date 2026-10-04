"""The extractors: the base class of custom ones, and the one of declared fields."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
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


def _accessor_name(model: type[Model], step: str) -> str:
    """Name the attribute of ``model`` that crosses the relation ``step``.

    A lookup path names a reverse relation by its query name; it is reached
    by its accessor.
    """
    relation = model._meta.get_field(step)
    if isinstance(relation, ForeignObjectRel):
        return relation.get_accessor_name() or step
    return step


def _related_instance(instance: Model, step: str) -> Model | None:
    """Cross the relation ``step`` of ``instance``, or ``None`` if unset.

    A missing reverse one-to-one raises instead of returning ``None``.
    """
    try:
        related: Model | None = getattr(instance, _accessor_name(type(instance), step))
    except ObjectDoesNotExist:
        return None
    return related


def _field_text(instance: Model, name: str) -> str:
    """Read the field ``name`` of ``instance`` as stripped text.

    ``name`` may be a lookup path, such as ``category__name``, to read a
    field of a related instance. A field with choices reads as its label.
    """
    *path, name = name.split(LOOKUP_SEP)
    for step in path:
        related = _related_instance(instance, step)
        if related is None:
            return ""
        instance = related
    value = getattr(instance, name)
    # Django adds get_<name>_display only to fields that have choices
    if getattr(instance._meta.get_field(name), "choices", None):
        value = getattr(instance, f"get_{name}_display")()
    return "" if value is None else str(value).strip()


def _document_text(field_texts: list[str]) -> str:
    """Join the non-empty ``field_texts``, in order."""
    return _FIELD_SEPARATOR.join(text for text in field_texts if text)


class DeclaredFieldsExtractor(BaseExtractor[Model]):
    """Build one document from the fields a model declares when registered."""

    def __init__(
        self,
        fields: tuple[str, ...],
        title_field: str | None,
        follow: tuple[str, ...] = (),
    ) -> None:
        """Read ``fields`` as the text, ``title_field`` (if any) as the title.

        The text of the relations in ``follow`` comes after the fields.
        """
        self.fields = fields
        self.title_field = title_field
        self.follow = follow
        # guessed once per related model; the registry builds a fresh
        # extractor for each run, so a redefined model is never served stale
        self._related_text_fields: dict[type[Model], list[str]] = {}

    def extract(self, instance: Model) -> NormalizedDocument | None:
        """Build the document of ``instance``, or nothing when it has no text.

        Its text is that of its fields, then of its followed relations.
        """
        field_texts = [_field_text(instance, name) for name in self.fields]
        followed_texts = [self._followed_text(instance, name) for name in self.follow]
        text = _document_text(field_texts + followed_texts)
        if not text:
            return None
        return self.build_document(
            instance, text=text, title=self._title(instance, field_texts)
        )

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
        return _field_text(instance, self.title_field)
