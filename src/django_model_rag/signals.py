"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import ForeignObjectRel, Model

from django_model_rag.documents import model_source_key
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import concrete_model_of, rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"
# The instance carries its followers from before the save to after it.
_PREVIOUS_FOLLOWERS_ATTRIBUTE = "_model_rag_previous_followers"
# It carries the followers pointing to it from before the delete to after it.
_FORWARD_FOLLOWERS_ATTRIBUTE = "_model_rag_forward_followers"
# Few enough that no query carries more variables than SQLite accepts.
_PKS_PER_QUERY = 500

# a registered model following an instance, and the primary key of its row
_Follower: TypeAlias = tuple[type[Model], Any]

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


def _followed_source_key(sender: type[Model], instance: Model) -> str:
    """Return the source key a failure to resync the followers of ``instance`` logs."""
    return model_source_key(concrete_model_of(sender), instance.pk)


def _models_of_the_row(sender: type[Model]) -> tuple[type[Model], ...]:
    """Return the models a row of ``sender`` is a row of, nearest first."""
    concrete_model = concrete_model_of(sender)
    # A multi-table child is a row of each of its parents too.
    return (concrete_model, *concrete_model._meta.get_parent_list())


def _registered_models(sender: type[Model]) -> list[type[Model]]:
    """Return the registered models whose groups ``sender``'s instances feed."""
    return [
        candidate
        for candidate in _models_of_the_row(sender)
        if rag.is_registered(candidate)
    ]


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


def _send_group(send: Callable[[], None], synced_rows: str) -> None:
    """Run ``send``, logging a failure that names the ``synced_rows``."""
    try:
        send()
    except Exception:
        # An error escaping a commit callback would break the commit.
        logger.exception("Syncing %s failed", synced_rows)


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
    """Fail before the INSERT or UPDATE if the save would sync a misconfigured output.

    A save syncs when ``sender`` feeds a registered model's group, or when a
    registered model follows it: an unregistered followed model is checked too.
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

    A save may move the row to other followers: the old ones change too. Only
    the followers the row points to can move; those pointing to the row through
    their own foreign key still do after the save, and are found then.
    """
    # An instance built with an existing primary key is "adding" yet saved as an
    # UPDATE: only a missing primary key means there is no row before the save.
    if raw or instance.pk is None or not _signals_enabled():
        return

    # A model followed only through foreign keys to it costs the save no query.
    if not _is_followed_through_reverse_relations(sender):
        return

    committed_instance = _committed_instance(sender, instance.pk)
    if committed_instance is not None:
        setattr(
            instance,
            _PREVIOUS_FOLLOWERS_ATTRIBUTE,
            _reverse_followers(sender, committed_instance),
        )


def sync_saved_instance(
    sender: type[Model],
    instance: Model,
    raw: bool = False,
    created: bool = False,
    **kwargs: Any,
) -> None:
    """Replace, once the transaction commits, the groups a saved instance changes.

    Those are the instance's own group if ``sender`` feeds a registered model,
    and the groups of its followers — the registered rows following it through
    a reverse relation, before and after the save — whether or not ``sender``
    is registered itself. The output configuration was checked before the
    save, by check_output_before_save.
    """
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    _schedule_commit_callbacks(registered_models, instance, _group_replacer)
    followers = _reverse_followers(sender, instance)
    # A row just created has no follower pointing to it yet.
    if not created:
        followers += _forward_followers(sender, instance)
    _schedule_follower_replacements(
        followers + instance.__dict__.pop(_PREVIOUS_FOLLOWERS_ATTRIBUTE, []),
        _followed_source_key(sender, instance),
    )


def _schedule_follower_replacements(
    followers: list[_Follower], followed_source_key: str
) -> None:
    """Replace, at the commit, the groups of ``followers``, one batch per model.

    A failure is logged with the ``followed_source_key`` of the row they follow.
    """
    pks_by_model: dict[type[Model], list[Any]] = {}
    # dict.fromkeys drops the duplicates and keeps the order.
    for follower_model, follower_pk in dict.fromkeys(followers):
        pks_by_model.setdefault(follower_model, []).append(follower_pk)
    for follower_model, follower_pks in pks_by_model.items():
        transaction.on_commit(
            _batch_replacer(follower_model, follower_pks, followed_source_key)
        )


def _batch_replacer(
    registered_model: type[Model], pks: list[Any], followed_source_key: str
) -> Callable[[], None]:
    """Return a commit callback replacing the groups of ``registered_model``'s rows."""

    def replace_groups_as_committed() -> None:
        for start in range(0, len(pks), _PKS_PER_QUERY):
            chunk = pks[start : start + _PKS_PER_QUERY]
            _send_group(
                partial(_replace_groups, registered_model, chunk),
                f"{registered_model._meta.label_lower} instances that follow "
                f"{followed_source_key}",
            )

    return replace_groups_as_committed


def _replace_groups(registered_model: type[Model], pks: list[Any]) -> None:
    """Send the committed groups of ``registered_model`` for ``pks`` to the output."""
    SyncPipeline(configured_output()).run_queryset(
        registered_model._base_manager.filter(pk__in=pks)
    )


def _reverse_followers(sender: type[Model], instance: Model) -> list[_Follower]:
    """Return the registered rows following ``instance`` through a reverse relation."""
    return [
        (registered_model, follower_pk)
        for registered_model in rag.registered_models()
        for relation in _followed_reverse_relations(registered_model, sender)
        for follower_pk in _follower_pks(registered_model, relation, instance)
    ]


def _forward_followers(sender: type[Model], instance: Model) -> list[_Follower]:
    """Return the registered rows following ``instance`` through foreign keys.

    They reach it through one foreign key of their own, or a chain of them.
    """
    return [
        (registered_model, follower_pk)
        for registered_model in rag.registered_models()
        for lookup, reached_model in _followed_foreign_key_lookups(
            registered_model, sender
        )
        for follower_pk in _pks_reaching(
            registered_model, lookup, _group_pk(instance, reached_model)
        )
    ]


def _followed_foreign_key_lookups(
    registered_model: type[Model], sender: type[Model]
) -> list[tuple[str, type[Model]]]:
    """Return the lookups through which ``registered_model`` reads ``sender``.

    Each crosses one foreign key or a chain of them, and comes with the model it
    reaches, one that ``sender``'s rows are rows of.
    """
    followed_models = _models_of_the_row(sender)
    return [
        (lookup, reached_model)
        for lookup, reached_model in rag.foreign_key_lookups(registered_model)
        # A foreign key may name a proxy: it reaches its concrete model's rows.
        if concrete_model_of(reached_model) in followed_models
    ]


def _pks_reaching(
    registered_model: type[Model], lookup: str, reached_pk: Any
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows reaching a row.

    Those rows reach, through ``lookup``, the row whose primary key is ``reached_pk``.
    """
    reaching_rows = registered_model._base_manager.filter(
        **{f"{lookup}__pk": reached_pk}
    )
    return list(reaching_rows.values_list("pk", flat=True))


def _is_followed(sender: type[Model]) -> bool:
    """Return whether a registered model follows ``sender``'s instances."""
    return any(
        _followed_reverse_relations(registered_model, sender)
        or _followed_foreign_key_lookups(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _is_followed_through_reverse_relations(sender: type[Model]) -> bool:
    """Return whether a registered model follows ``sender`` by a reverse relation."""
    return any(
        _followed_reverse_relations(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _follower_pks(
    registered_model: type[Model], relation: ForeignObjectRel, instance: Model
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows ``instance`` points to.

    ``instance`` points to them through the foreign key behind ``relation``.
    """
    foreign_key = relation.field
    # The instance's columns, matched to those of the follower row they name.
    follower_lookup = {
        target.attname: getattr(instance, local.attname)
        for local, target in zip(
            foreign_key.local_related_fields,
            foreign_key.foreign_related_fields,
            strict=True,
        )
    }
    if None in follower_lookup.values():
        # A null foreign key points to no follower.
        return []
    target_field = foreign_key.foreign_related_fields[0]
    if target_field.primary_key and len(follower_lookup) == 1:
        # The common case costs the save no query: the value already is the key.
        return [follower_lookup[target_field.attname]]

    # A foreign key with a to_field, or over several columns, holds other
    # columns: the group is named after the primary key, which only the
    # follower row knows.
    follower_rows = registered_model._base_manager.filter(**follower_lookup)
    return list(follower_rows.values_list("pk", flat=True))


def _followed_reverse_relations(
    registered_model: type[Model], sender: type[Model]
) -> list[ForeignObjectRel]:
    """Return the reverse relations to ``sender`` that ``registered_model`` follows."""
    followed_models = _models_of_the_row(sender)
    return [
        relation
        for relation in rag.followed_reverse_relations(registered_model)
        if relation.related_model in followed_models
    ]


def _group_replacer(registered_model: type[Model], pk: Any) -> Callable[[], None]:
    """Return a commit callback replacing the group of a saved instance's row.

    ``pk`` is the saved instance's primary key as a ``registered_model`` row.
    """

    def replace_group_as_committed() -> None:
        _send_group(
            lambda: _replace_group(registered_model, pk),
            model_source_key(registered_model, pk),
        )

    return replace_group_as_committed


def remember_followers_before_delete(
    sender: type[Model], instance: Model, **kwargs: Any
) -> None:
    """Keep the followers pointing to the row, for the commit to replace.

    Django may null their foreign key before the row goes: by post_delete they
    no longer point to it.
    """
    if not _signals_enabled():
        return

    setattr(
        instance,
        _FORWARD_FOLLOWERS_ATTRIBUTE,
        _forward_followers(sender, instance),
    )


def sync_deleted_instance(sender: type[Model], instance: Model, **kwargs: Any) -> None:
    """Empty the group of a deleted instance, and replace those following it.

    Both happen once the transaction commits.
    """
    if not _signals_enabled():
        return

    # Django sends post_delete for each multi-table parent too, under its own
    # sender: the nearest registered model is the only group to empty here.
    nearest_registered_model_only = _registered_models(sender)[:1]
    followers = _reverse_followers(sender, instance) + instance.__dict__.pop(
        _FORWARD_FOLLOWERS_ATTRIBUTE, []
    )
    if not (nearest_registered_model_only or followers):
        return

    # Fail at the delete, not at the commit, if the output is misconfigured.
    check_output_configuration()
    _schedule_commit_callbacks(nearest_registered_model_only, instance, _group_emptier)
    _schedule_follower_replacements(followers, _followed_source_key(sender, instance))


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
