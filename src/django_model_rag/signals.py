"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable
from functools import partial, reduce
from operator import or_
from typing import Any, TypeAlias, cast

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import (
    ForeignObject,
    ForeignObjectRel,
    ManyToManyField,
    ManyToManyRel,
    Model,
    Q,
    QuerySet,
)

from django_model_rag.documents import model_source_key
from django_model_rag.extractors import query_name, through_key_to_parent
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import concrete_model_of, rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"
_CHANGING_ACTIONS = frozenset({"post_add", "post_remove", "post_clear"})
_BEFORE_CLEAR = "pre_clear"
_BEFORE_WRITE = frozenset({"pre_add", "pre_remove", _BEFORE_CLEAR})
# The instance carries the primary keys of the rows its links reach, from before
# a clear.
_CLEARED_PKS_ATTRIBUTE = "_model_rag_cleared_pks"
# The instance carries its followers from before the save to after it.
_PREVIOUS_FOLLOWERS_ATTRIBUTE = "_model_rag_previous_followers"
# It carries the followers pointing to it from before the delete to after it.
_REACHING_FOLLOWERS_ATTRIBUTE = "_model_rag_reaching_followers"
# Few enough that no query carries more variables than SQLite accepts.
_PKS_PER_QUERY = 500

# a registered model following an instance, and the primary key of its row
_Follower: TypeAlias = tuple[type[Model], Any]

logger = logging.getLogger("django_model_rag")


def _signals_enabled() -> bool:
    """Return whether the signal receivers sync anything, as the settings say."""
    return bool(getattr(settings, _SIGNALS_SETTING, True))


def _committed_instance(rows: QuerySet[Model], pk: Any) -> Model | None:
    """Return the row of ``rows`` with primary key ``pk`` as committed, if any."""
    try:
        return rows.get(pk=pk)
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
    # run_instance() reloads the row itself: only whether it still exists, and
    # its primary key as stored, are needed here.
    committed_instance = _committed_instance(
        registered_model._base_manager.only("pk"), pk
    )
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

    A save may move the row to other followers: the old ones change too. It
    may point the row to other followers, or change the columns other
    followers name it by.
    """
    # An instance built with an existing primary key is "adding" yet saved as an
    # UPDATE: only a missing primary key means there is no row before the save.
    if raw or instance.pk is None or not _signals_enabled():
        return

    # A model followed by nothing costs the save no query.
    if not _is_followed(sender):
        return

    # Followers are looked up by the row's primary key, joined against the
    # columns still in the database: pre_save runs before the UPDATE, so they
    # are those of the row as committed, even by columns the save changes. At
    # the commit, they are looked up again from the row as saved: only the
    # lookups made here find the followers it had before the save.
    setattr(
        instance,
        _PREVIOUS_FOLLOWERS_ATTRIBUTE,
        _committed_followers(sender, instance),
    )


def _committed_followers(sender: type[Model], instance: Model) -> list[_Follower]:
    """Return the registered rows following ``instance``'s row as committed.

    Both those reaching it through foreign keys and those it points to through
    a reverse relation are looked up by its primary key, joined against the
    columns in the database: before the save, they are those of the row as it
    was, even if the save changes them; at the commit, those of the row then,
    even through a write sending no signal.
    """
    return _followers_reaching(sender, instance) + _followers_pointed_to(
        sender, instance
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
    and the groups of its followers — the registered rows following it, before
    the save and after it — whether or not ``sender`` is registered itself.
    On an update, the followers from before the save were found at pre_save,
    and those of the row as saved are only found by a lookup at the commit. On
    a create, the followers the new row points to are read here, and looked up
    again at the commit if there are any. The output configuration was checked
    before the save, by check_output_before_save.
    """
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    _schedule_commit_callbacks(registered_models, instance, _group_replacer)
    # On an update, only the followers from before the save: those the row as
    # saved points to are left to the commit's lookup.
    followers_at_save = instance.__dict__.pop(_PREVIOUS_FOLLOWERS_ATTRIBUTE, [])
    is_followed = _is_followed(sender)
    # On a create, the followers the new row points to are read from the
    # instance, with no query in the common case: whether there are any decides
    # if the commit looks them up again. At post_save the instance is the row
    # as saved: a change left unsaved in memory afterwards is not followed.
    if created:
        followers_at_save = _reverse_followers(sender, instance) + followers_at_save
    # A row just created has no follower pointing to it at its save, and
    # costs the commit no lookup unless it points to a follower.
    if (followers_at_save or not created) and is_followed:
        # Rows may also be attached to it, or it to other rows, before the
        # commit, by a write that sends no signal: its followers are looked up
        # then.
        transaction.on_commit(
            partial(
                _replace_followers_as_committed,
                sender,
                instance,
                followers_at_save,
            )
        )
        return

    _schedule_follower_replacements(
        followers_at_save, _followed_source_key(sender, instance)
    )


def sync_changed_relation(
    instance: Model,
    action: str,
    model: type[Model],
    pk_set: set[Any] | None = None,
    **kwargs: Any,
) -> None:
    """Replace, once the transaction commits, the groups a change of links alters.

    Those are the group of the instance, and the groups of the registered rows
    that ``pk_set`` names, from either side of the links.
    """
    if not _signals_enabled():
        return

    through = kwargs["sender"]
    reverse = kwargs["reverse"]
    instance_side, model_side = _link_relations(through, instance, model, reverse)
    instance_lookup, model_lookup = query_name(instance_side), query_name(model_side)
    if action in _BEFORE_WRITE:
        # Keys left by a clear that failed after pre_clear are not this change's.
        instance.__dict__.pop(_CLEARED_PKS_ATTRIBUTE, None)
        _check_output_before_write(instance, model, instance_lookup, model_lookup)
        if action == _BEFORE_CLEAR:
            _remember_cleared_pks(instance, model, model_lookup)
        return

    if action not in _CHANGING_ACTIONS:
        return

    _schedule_commit_callbacks(
        _registered_models_following(type(instance), instance_lookup),
        instance,
        _group_replacer,
    )
    if _reaches_registered_rows(model, model_lookup):
        # Only a clear leaves primary keys behind, found before it: pk_set is
        # None then.
        changed_pks = _primary_keys_named(
            _through_key(model_side), model, pk_set or set()
        ) | instance.__dict__.pop(_CLEARED_PKS_ATTRIBUTE, set())
        _schedule_follower_replacements(
            [(model, pk) for pk in changed_pks],
            _followed_source_key(type(instance), instance),
        )


def _check_output_before_write(
    instance: Model, model: type[Model], instance_lookup: str, model_lookup: str
) -> None:
    """Fail before the join rows are written or deleted if the sync would fail.

    ``instance_lookup`` and ``model_lookup`` are the lookups the instance's
    side and ``model``'s read the links with. In autocommit the join rows are
    committed as soon as they are written: after the change, a failing check
    would come too late to undo it.
    """
    if _registered_models_following(type(instance), instance_lookup) or (
        _reaches_registered_rows(model, model_lookup)
    ):
        check_output_configuration()


def _registered_models_following(
    sender: type[Model], link_lookup: str
) -> list[type[Model]]:
    """Return the registered models of ``sender`` following ``link_lookup``.

    ``link_lookup`` is the lookup ``sender``'s side reads the changed links with.
    """
    return [
        registered_model
        for registered_model in _registered_models(sender)
        if _follows_lookup(registered_model, link_lookup)
    ]


def _follows_lookup(registered_model: type[Model], lookup: str) -> bool:
    """Return whether ``registered_model`` reads its rows' relations by ``lookup``."""
    return bool(_followed_lookups_named(registered_model, lookup))


def _followed_lookups_named(
    registered_model: type[Model], lookup: str
) -> list[tuple[str, type[Model]]]:
    """Return ``registered_model``'s followed lookups named ``lookup``.

    Each comes with the model it reaches.
    """
    return [
        (followed_lookup, reached_model)
        for followed_lookup, reached_model in rag.foreign_key_lookups(registered_model)
        if followed_lookup == lookup
    ]


def _link_relations(
    through: type[Model], instance: Model, model: type[Model], reverse: bool
) -> tuple[
    "ManyToManyField[Any, Any] | ManyToManyRel",
    "ManyToManyField[Any, Any] | ManyToManyRel",
]:
    """Return the relations the instance's side, then ``model``'s, reads links with.

    Those are the links of ``through``. ``reverse``, as m2m_changed sends it,
    is False when the instance's model declares the many-to-many: on a
    many-to-many from a model to itself, it tells the two sides apart.
    """
    declaring_model = model if reverse else type(instance)
    # m2m_changed comes from a many-to-many of the declaring model, inherited
    # or not, and each many-to-many has a through model of its own.
    field = next(
        field
        for field in declaring_model._meta.many_to_many
        if field.remote_field.through is through
    )
    return (field.remote_field, field) if reverse else (field, field.remote_field)


def _through_key(
    relation: "ManyToManyField[Any, Any] | ManyToManyRel",
) -> "ForeignObject[Any, Any]":
    """Return the through model's foreign key to the side ``relation`` is read from."""
    # A through model reaches each side of its many-to-many by a foreign key.
    return cast("ForeignObject[Any, Any]", through_key_to_parent(relation))


def _primary_keys_named(
    key_to_model: "ForeignObject[Any, Any]", model: type[Model], keys: set[Any]
) -> set[Any]:
    """Return the primary keys of the ``model`` rows that ``keys`` name.

    The through model's foreign key to ``model``, ``key_to_model``, may point to
    a unique column other than its primary key, and ``keys`` are then values of
    that column.
    """
    target_field = key_to_model.target_field
    if target_field.primary_key:
        return keys

    return set(
        model._base_manager.filter(**{f"{target_field.name}__in": keys}).values_list(
            "pk", flat=True
        )
    )


def _reaches_registered_rows(model: type[Model], link_lookup: str) -> bool:
    """Return whether a links change reaches rows following the links.

    ``link_lookup`` is the lookup ``model``'s side reads the changed links with.
    """
    return rag.is_registered(model) and _follows_lookup(model, link_lookup)


def _remember_cleared_pks(
    instance: Model, model: type[Model], link_lookup: str
) -> None:
    """Keep the primary keys of the registered rows a clear is about to unlink.

    Django sends no primary keys with the clear: they can only be found before
    it. They are those of the ``model`` rows whose ``link_lookup``, the lookup
    they read the links with, reaches the instance.
    """
    if not _reaches_registered_rows(model, link_lookup):
        return

    # The lookup reaches a multi-table child by the row of the parent holding
    # the links.
    linked_pks = _pks_reaching(
        model,
        [
            (lookup, _group_pk(instance, reached_model))
            for lookup, reached_model in _followed_lookups_named(model, link_lookup)
        ],
    )
    setattr(instance, _CLEARED_PKS_ATTRIBUTE, set(linked_pks))


def _replace_followers_as_committed(
    sender: type[Model],
    instance: Model,
    followers_at_save: list[_Follower],
) -> None:
    """Replace the groups of ``followers_at_save`` and of the rows following it now.

    On a create, ``followers_at_save`` are the followers the new row points
    to; on an update, only those it had before the save. The rows following
    it now are looked up from the row as committed: on an update, those the
    save pointed it to; rows attached after the save, in the same transaction;
    and the rows a write sending no signal has since pointed it to. If looking
    them up fails, ``followers_at_save`` are still replaced, but on an update a
    follower the row was moved to is not.
    """
    followed_source_key = _followed_source_key(sender, instance)
    try:
        followers = _committed_followers(sender, instance) + followers_at_save
    except Exception:
        # An error escaping a commit callback would break the commit.
        logger.exception("Looking up the followers of %s failed", followed_source_key)
        followers = followers_at_save

    for replace_groups in _follower_replacers(followers, followed_source_key):
        replace_groups()


def _schedule_follower_replacements(
    followers: list[_Follower], followed_source_key: str
) -> None:
    """Replace, at the commit, the groups of ``followers``, one batch per model.

    A failure is logged with the ``followed_source_key`` of the row they follow.
    """
    for replace_groups in _follower_replacers(followers, followed_source_key):
        transaction.on_commit(replace_groups)


def _follower_replacers(
    followers: list[_Follower], followed_source_key: str
) -> list[Callable[[], None]]:
    """Return one callback per model replacing the groups of its ``followers``."""
    pks_by_model: dict[type[Model], list[Any]] = {}
    # dict.fromkeys drops the duplicates and keeps the order.
    for follower_model, follower_pk in dict.fromkeys(followers):
        pks_by_model.setdefault(follower_model, []).append(follower_pk)
    return [
        _batch_replacer(follower_model, follower_pks, followed_source_key)
        for follower_model, follower_pks in pks_by_model.items()
    ]


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


def _followers_reaching(sender: type[Model], instance: Model) -> list[_Follower]:
    """Return the registered rows following ``instance`` through a lookup.

    They reach it through one foreign key of their own, a chain of them, or a
    many-to-many.
    """
    return _followers_looked_up(_followed_lookups, sender, instance)


def _followers_pointed_to(sender: type[Model], instance: Model) -> list[_Follower]:
    """Return the registered rows ``instance``'s row points to, looked up.

    It points to them through the foreign key behind a reverse relation they
    follow; the lookup reads that key as stored, not as ``instance`` holds it.
    """
    if instance.pk is None:
        # A row deleted since its save has lost its primary key, but a
        # multi-table child keeps its parent link: looking it up would query a
        # stale id, which a row inserted since may have taken.
        return []

    return _followers_looked_up(_followed_reverse_lookups, sender, instance)


def _followers_looked_up(
    followed_lookups: Callable[
        [type[Model], type[Model]], list[tuple[str, type[Model]]]
    ],
    sender: type[Model],
    instance: Model,
) -> list[_Follower]:
    """Return the registered rows reaching ``instance``'s row by a followed lookup.

    ``followed_lookups`` gives, for a registered model and ``sender``, each
    lookup it follows with the model it reaches, one ``sender``'s rows are
    rows of.
    """
    return [
        (registered_model, follower_pk)
        for registered_model in rag.registered_models()
        for follower_pk in _pks_reaching(
            registered_model,
            [
                (lookup, _group_pk(instance, reached_model))
                for lookup, reached_model in followed_lookups(registered_model, sender)
            ],
        )
    ]


def _followed_reverse_lookups(
    registered_model: type[Model], sender: type[Model]
) -> list[tuple[str, type[Model]]]:
    """Return the lookups crossing the followed reverse relations to ``sender``.

    They are ``registered_model``'s. Each comes with the model the relation
    starts from, which may be a multi-table parent of ``sender``, whose row is
    reached by the parent link.
    """
    return [
        (query_name(relation), relation.related_model)
        for relation in _followed_reverse_relations(registered_model, sender)
    ]


def _followed_lookups(
    registered_model: type[Model], sender: type[Model]
) -> list[tuple[str, type[Model]]]:
    """Return the lookups through which ``registered_model`` reads ``sender``.

    Each crosses one foreign key, a chain of them, or a many-to-many, and comes
    with the model it reaches, one that ``sender``'s rows are rows of.
    """
    followed_models = _models_of_the_row(sender)
    return [
        (lookup, reached_model)
        for lookup, reached_model in rag.foreign_key_lookups(registered_model)
        # A foreign key may name a proxy: it reaches its concrete model's rows.
        if concrete_model_of(reached_model) in followed_models
    ]


def _pks_reaching(
    registered_model: type[Model], lookups_to_reached_pks: list[tuple[str, Any]]
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows reaching a row.

    Each of ``lookups_to_reached_pks`` pairs a lookup with the primary key of
    the row it must reach; a row reaching any of them is returned. One query
    crosses all the lookups.
    """
    rows = registered_model._base_manager.all()
    rows_through_lookups = [
        rows.filter(**{f"{lookup}__pk": reached_pk})
        for lookup, reached_pk in lookups_to_reached_pks
        # A row deleted since its save has lost its primary key: filtering on
        # None would match the rows whose foreign key is null.
        if reached_pk is not None
    ]
    if not rows_through_lookups:
        return []

    reaching_rows = _rows_in_any(rows, rows_through_lookups)
    return list(reaching_rows.values_list("pk", flat=True))


def _rows_in_any(
    rows: QuerySet[Model], subsets: list[QuerySet[Model]]
) -> QuerySet[Model]:
    """Return the ``rows`` in any of ``subsets``, each a filter of ``rows``."""
    if len(subsets) == 1:
        return subsets[0]

    # A subquery per subset keeps each as selective as a query of its own:
    # ORed in one filter(), lookups crossing multi-valued relations would join
    # them all, with the OR across the joins, which no index serves.
    return rows.filter(
        reduce(or_, (Q(pk__in=subset.values("pk")) for subset in subsets))
    )


def _is_followed(sender: type[Model]) -> bool:
    """Return whether a registered model follows ``sender``'s instances."""
    return _is_followed_through_lookups(
        sender
    ) or _is_followed_through_reverse_relations(sender)


def _is_followed_through_reverse_relations(sender: type[Model]) -> bool:
    """Return whether a registered model follows ``sender``'s rows in reverse."""
    return any(
        _followed_reverse_relations(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _is_followed_through_lookups(sender: type[Model]) -> bool:
    """Return whether a registered model reads ``sender``'s rows by a lookup."""
    return any(
        _followed_lookups(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _follower_pks(
    registered_model: type[Model], relation: ForeignObjectRel, instance: Model
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows ``instance`` points to.

    ``instance`` points to them through the foreign key behind ``relation``.
    """
    foreign_key = relation.field
    follower_lookup = _follower_lookup(foreign_key, instance)
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


def _follower_lookup(
    foreign_key: "ForeignObject[Any, Any]", instance: Model
) -> dict[str, Any]:
    """Return the follower row's columns, each with the value ``instance`` holds."""
    return dict(
        zip(
            (target.attname for target in foreign_key.foreign_related_fields),
            foreign_key.get_local_related_value(instance),
            strict=True,
        )
    )


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
        _REACHING_FOLLOWERS_ATTRIBUTE,
        _followers_reaching(sender, instance),
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
        _REACHING_FOLLOWERS_ATTRIBUTE, []
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
