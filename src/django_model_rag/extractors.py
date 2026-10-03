"""The extractors: the base class of custom ones, and the one of declared fields."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import CharField, Field, Manager, Model, TextField

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


def _field_text(instance: Model, name: str) -> str:
    """Read the field ``name`` of ``instance`` as stripped text.

    A field with choices reads as its label.
    """
    value = getattr(instance, name)
    # Django adds get_<name>_display only to fields that have choices
    if getattr(instance._meta.get_field(name), "choices", None):
        value = getattr(instance, f"get_{name}_display")()
    return "" if value is None else str(value).strip()


def _document_text(field_texts: list[str]) -> str:
    """Join the non-empty ``field_texts``, in order."""
    return _FIELD_SEPARATOR.join(text for text in field_texts if text)


def _related_text(related: Model | Manager[Model] | None) -> str:
    """Join the texts of the text fields of ``related``, title-like ones first.

    ``related`` may be the manager of a reverse relation: its objects follow
    one another, in primary key order.
    """
    if related is None:
        return ""
    if isinstance(related, Manager):
        return _document_text([_related_text(item) for item in related.order_by("pk")])
    return _document_text(
        [_field_text(related, name) for name in text_fields(type(related))]
    )


def _followed_text(instance: Model, name: str) -> str:
    """Read the text of the relation ``name`` of ``instance``.

    A reverse one-to-one without an object raises, where it has no text.
    """
    try:
        return _related_text(getattr(instance, name))
    except ObjectDoesNotExist:
        return ""


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

    def extract(self, instance: Model) -> NormalizedDocument | None:
        """Build the document of ``instance``, or nothing when its fields are blank."""
        field_texts = [_field_text(instance, name) for name in self.fields]
        followed_texts = [_followed_text(instance, name) for name in self.follow]
        text = _document_text(field_texts + followed_texts)
        if not text:
            return None
        return self.build_document(
            instance, text=text, title=self._title(instance, field_texts)
        )

    def _title(self, instance: Model, field_texts: list[str]) -> str:
        """Take the text of the title field, or the first of ``field_texts``.

        ``field_texts`` are the texts of the declared fields, in order: a title
        field among them is not read a second time.
        """
        if self.title_field is None:
            return field_texts[0]
        if self.title_field in self.fields:
            return field_texts[self.fields.index(self.title_field)]
        return _field_text(instance, self.title_field)
