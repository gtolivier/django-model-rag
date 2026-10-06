"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable
from typing import Any

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import ForeignObjectRel, Model

from django_model_rag.documents import model_source_key
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"
# The instance carries its followers from before the save to after it.
_PREVIOUS_FOLLOWERS_ATTRIBUTE = "_model_rag_previous_followers"

logger = logging.getLogger("django_model_rag")


def _signals_enabled() -> bool:
    """Return whether the signal receivers sync anything, as the settings say."""
    return bool(getattr(settings, _SIGNALS_SETTING, True))


def _committed_instance(model: type[Model], pk: Any) -> Model | None:
    """Return the instance of ``model`` with primary key ``pk`` as committed, if any."""
    try:
        return model._base_manager.get(pk=pk)
    except ObjectDoesNotExist:
        return None


def _registered_models(sender: type[Model]) -> list[type[Model]]:
    """Return the registered models whose groups ``sender``'s instances feed."""
    # A proxy sends signals under its own sender: its group is the concrete model's.
    concrete_model = sender._meta.concrete_model
    if concrete_model is None:
        return []
    # A multi-table child feeds the group of its registered parents too.
    candidates: tuple[type[Model], ...] = (
        concrete_model,
        *concrete_model._meta.get_parent_list(),
    )
    return [candidate for candidate in candidates if rag.is_registered(candidate)]


def _schedule_commit_callbacks(
    registered_models: list[type[Model]],
    instance: Model,
    build_callback: Callable[[type[Model], Any], Callable[[], None]],
) -> None:
    """Run, at the commit, the callback ``build_callback`` returns per registered model.

    Each callback gets the primary key of the group as it is now: delete()
    clears the primary key of the instance before the commit.
    """
    for registered_model in registered_models:
        pk = _group_pk(instance, registered_model)
        transaction.on_commit(build_callback(registered_model, pk))


def _group_pk(instance: Model, registered_model: type[Model]) -> Any:
    """Return the primary key of the ``registered_model`` row ``instance`` is."""
    # A multi-table child may have a primary key of its own: the parent row is
    # reached by the parent link.
    link = instance._meta.get_ancestor_link(registered_model)
    if link is None:
        return instance.pk
    return getattr(instance, link.attname)


def _send_group(send: Callable[[], None], source_key: str) -> None:
    """Run ``send``, logging a failure with the ``source_key`` of the group."""
    try:
        send()
    except Exception:
        # An error escaping a commit callback would break the commit.
        logger.exception("Syncing %s failed", source_key)


def _replace_group(registered_model: type[Model], pk: Any) -> None:
    """Send the committed group of ``registered_model`` for ``pk`` to the output."""
    committed_instance = _committed_instance(registered_model, pk)
    # Deleted since the save: the delete's own callback sends the empty group.
    if committed_instance is None:
        return

    SyncPipeline(configured_output()).run_instance(committed_instance)


def check_output_before_save(
    sender: type[Model], raw: bool = False, **kwargs: Any
) -> None:
    """Fail before the INSERT or UPDATE if the output is misconfigured.

    In autocommit the row is committed as soon as it is written: after the
    save, a failing check would come too late to keep the row out.
    """
    if raw or not _signals_enabled():
        return

    if _registered_models(sender) or _is_followed(sender):
        check_output_configuration()


def remember_followers_before_save(
    sender: type[Model], instance: Model, raw: bool = False, **kwargs: Any
) -> None:
    """Keep the followers the row had before the save, for the commit to replace.

    A save may move the row to other followed rows: the old ones change too.
    """
    if raw or instance._state.adding or not _signals_enabled():
        return

    # A model nothing follows costs the save no query.
    if not _is_followed(sender):
        return

    committed_instance = _committed_instance(sender, instance.pk)
    if committed_instance is not None:
        setattr(
            instance,
            _PREVIOUS_FOLLOWERS_ATTRIBUTE,
            _followers(sender, committed_instance),
        )


def sync_saved_instance(
    sender: type[Model], instance: Model, raw: bool = False, **kwargs: Any
) -> None:
    """Replace the group of a saved registered instance once its transaction commits.

    The output configuration was checked before the save, by
    check_output_before_save.
    """
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    _schedule_commit_callbacks(registered_models, instance, _group_replacer)
    _schedule_follower_replacements(_followers(sender, instance))
    _schedule_follower_replacements(
        getattr(instance, _PREVIOUS_FOLLOWERS_ATTRIBUTE, [])
    )


def _schedule_follower_replacements(followers: list[tuple[type[Model], Any]]) -> None:
    """Replace, at the commit, the group of each of ``followers``."""
    for followed_by, followed_pk in followers:
        transaction.on_commit(_group_replacer(followed_by, followed_pk))


def _followers(sender: type[Model], instance: Model) -> list[tuple[type[Model], Any]]:
    """Return the registered models following ``instance``, with their primary keys.

    Only the reverse foreign keys are looked at.
    """
    return [
        (registered_model, followed_pk)
        for registered_model in rag.registered_models()
        for relation in _followed_reverse_relations(registered_model, sender)
        for followed_pk in _followed_pks(registered_model, relation, instance)
    ]


def _is_followed(sender: type[Model]) -> bool:
    """Return whether a registered model follows ``sender``'s instances."""
    return any(
        _followed_reverse_relations(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _followed_pks(
    registered_model: type[Model], relation: ForeignObjectRel, instance: Model
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows ``instance`` points to.

    ``instance`` points to them through the foreign key behind ``relation``.
    """
    foreign_key = relation.field
    target_field = foreign_key.foreign_related_fields[0]
    target_value = getattr(instance, foreign_key.attname)
    if target_field.primary_key:
        # The common case costs the save no query: the value already is the key.
        return [target_value]

    # A foreign key with a to_field holds another unique column: the group is
    # named after the primary key, which only the followed row knows.
    followed_rows = registered_model._base_manager.filter(
        **{target_field.attname: target_value}
    )
    return list(followed_rows.values_list("pk", flat=True))


def _followed_reverse_relations(
    registered_model: type[Model], sender: type[Model]
) -> list[ForeignObjectRel]:
    """Return the reverse relations to ``sender`` that ``registered_model`` follows."""
    return [
        relation
        for relation in rag.followed_reverse_relations(registered_model)
        if relation.related_model is sender
    ]


def _group_replacer(registered_model: type[Model], saved_pk: Any) -> Callable[[], None]:
    """Return a commit callback replacing the group of a saved instance."""

    def replace_group_as_committed() -> None:
        _send_group(
            lambda: _replace_group(registered_model, saved_pk),
            model_source_key(registered_model, saved_pk),
        )

    return replace_group_as_committed


def sync_deleted_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Empty the group of a deleted instance, and replace those following it.

    Both happen once the transaction commits.
    """
    if not _signals_enabled():
        return

    # Django sends post_delete for each multi-table parent too, under its own
    # sender: the nearest registered model is the only group to empty here.
    nearest_registered_model_only = _registered_models(sender)[:1]
    followers = _followers(sender, instance)
    if not (nearest_registered_model_only or followers):
        return

    # Fail at the delete, not at the commit, if the output is misconfigured.
    check_output_configuration()
    _schedule_commit_callbacks(nearest_registered_model_only, instance, _group_emptier)
    _schedule_follower_replacements(followers)


def _group_emptier(
    registered_model: type[Model], deleted_pk: Any
) -> Callable[[], None]:
    """Return a commit callback emptying the group of a deleted instance."""

    def replace_group_with_an_empty_one() -> None:
        # The row is gone: the empty group goes straight to the output, with no
        # extractor involved.
        source_key = model_source_key(registered_model, deleted_pk)
        _send_group(lambda: configured_output().replace({source_key: []}), source_key)

    return replace_group_with_an_empty_one
