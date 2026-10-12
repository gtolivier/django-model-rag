"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from functools import partial, reduce
from operator import or_
from typing import Any, TypeAlias

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
from django_model_rag.extractors import query_name
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import concrete_model_of, rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"
_BEFORE_UNLINK = frozenset({"pre_remove", "pre_clear"})
_AFTER_UNLINK = frozenset({"post_remove", "post_clear"})
_CHANGING_ACTIONS = frozenset({"post_add"}) | _AFTER_UNLINK
_BEFORE_WRITE = frozenset({"pre_add"}) | _BEFORE_UNLINK
# The instance carries its followers from before the save to after it.
_PREVIOUS_FOLLOWERS_ATTRIBUTE = "_model_rag_previous_followers"
# It carries the followers pointing to it from before the delete to after it.
_REACHING_FOLLOWERS_ATTRIBUTE = "_model_rag_reaching_followers"
# Few enough that no query carries more variables than SQLite accepts.
_PKS_PER_QUERY = 500

# a registered model following an instance, and the primary key of its row
_Follower: TypeAlias = tuple[type[Model], Any]
# a relation one side of a many-to-many reads its links with
_LinkRelation: TypeAlias = "ManyToManyField[Any, Any] | ManyToManyRel"

logger = logging.getLogger("django_model_rag")

# The through model whose rows a remove() or clear() is deleting: from its
# pre_ m2m_changed to its post_ one.
_through_being_unlinked: ContextVar[type[Model] | None] = ContextVar(
    "model_rag_through_being_unlinked", default=None
)


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

    A save may move the row to other followers: the old ones change too.
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
        _followers_pointed_to(sender, instance),
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
    again at the commit if there are any, if a registered model reads
    ``sender`` by a lookup, or if ``sender`` is the through model of a followed
    many-to-many. The output configuration was checked before the save, by
    check_output_before_save.
    """
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    _schedule_commit_callbacks(registered_models, instance, _group_replacer)
    # On an update, only the followers from before the save: those the row as
    # saved points to are left to the commit's lookup.
    followers_at_save = instance.__dict__.pop(_PREVIOUS_FOLLOWERS_ATTRIBUTE, [])
    # A model followed by nothing has no follower before the save or after it.
    if not _is_followed(sender):
        return

    # On a create, the followers the new row points to are read from the
    # instance, with no query in the common case. At post_save the instance is
    # the row as saved: a change left unsaved in memory afterwards is not
    # followed.
    if created:
        followers_at_save = _reverse_followers(sender, instance) + followers_at_save
        # A row just created has no follower pointing to it at its save.
        # Followed only in reverse, it costs the commit no lookup unless it
        # points to a follower; read by a lookup, or linking rows of a followed
        # many-to-many, it may be reached by rows written since.
        if not (
            followers_at_save
            or _is_followed_through_lookups(sender)
            or _is_through_of_followed(sender)
        ):
            return

    # Rows may also be attached to it, or it to other rows, before the commit,
    # by a write that sends no signal: its followers are looked up then.
    _resync_followers_at_commit(
        _followed_source_key(sender, instance),
        partial(_followers_as_committed, sender, instance, followers_at_save),
    )


def sync_changed_relation(
    instance: Model,
    action: str,
    model: type[Model],
    **kwargs: Any,
) -> None:
    """Replace, once the transaction commits, the groups a change of links alters.

    The instance is replaced for each registered model of it following the
    lookup its side reads the links with, and every row of the registered
    models reading the instance's model is replaced too, whichever row it
    reaches. On a many-to-many from a model to itself, the instance alone is
    replaced when only its side is followed; otherwise the other side reads
    the links too, so every row reading the model is resynced, the
    instance's own included. From a remove() or clear() until it ends, the
    through model whose rows it deletes is noted, for sync_deleted_instance.
    """
    if not _signals_enabled():
        return

    through = kwargs["sender"]
    _note_unlinking(action, through)
    reverse = kwargs["reverse"]
    relations = _link_relations(through, instance, model, reverse)
    if relations is None:
        return

    instance_side, model_side = relations
    model_link_lookup = query_name(model_side)
    following_models = _registered_models_following(
        type(instance), query_name(instance_side)
    )
    model_lookups = _registered_lookups_named(model, model_link_lookup)
    if action in _BEFORE_WRITE:
        _check_output_before_write(following_models, model_lookups)
        return

    if action not in _CHANGING_ACTIONS:
        return

    if model is not type(instance):
        _schedule_commit_callbacks(following_models, instance, _group_replacer)
        _resync_every_row_reading(instance)
    elif following_models and not _registered_subclasses_following(
        model, model_link_lookup
    ):
        # From a model to itself, read by the instance's side alone.
        _schedule_commit_callbacks(following_models, instance, _group_replacer)
    else:
        # From a model to itself, read by the other side too: the instance's
        # rows are among the rows resynced.
        _resync_every_row_reading(instance)


def _resync_every_row_reading(instance: Model) -> None:
    """Replace, at the commit, every row of the registered models reading ``instance``.

    Those are the models reading the instance's model, whichever row they reach.
    """
    _schedule_follower_replacements(
        _every_row_reading(type(instance)),
        _followed_source_key(type(instance), instance),
    )


def _note_unlinking(action: str, through: type[Model]) -> None:
    """Remember the through model a remove() or clear() deletes the rows of."""
    if action in _BEFORE_UNLINK:
        _through_being_unlinked.set(concrete_model_of(through))
    elif action in _AFTER_UNLINK:
        _through_being_unlinked.set(None)


def _check_output_before_write(
    following_models: list[type[Model]], model_lookups: list[tuple[str, type[Model]]]
) -> None:
    """Fail before the join rows are written or deleted if the sync would fail.

    ``following_models`` are the instance's registered models following the
    links, ``model_lookups`` the lookups by which the other side's registered
    rows follow them. In autocommit the join rows are committed as soon as
    they are written: after the change, a failing check would come too late
    to undo it.
    """
    if following_models or model_lookups:
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


def _registered_subclasses_following(
    model: type[Model], link_lookup: str
) -> list[type[Model]]:
    """Return the registered subclasses of ``model`` following ``link_lookup``.

    Their rows are ``model`` rows: ``model`` itself, or a multi-table child of
    it, sharing its keys. ``link_lookup`` is the lookup ``model``'s side reads
    the changed links with.
    """
    return [
        registered_model
        for registered_model in rag.registered_models()
        # A many-to-many may name a proxy: it reaches its concrete model's rows.
        if concrete_model_of(model) in _models_of_the_row(registered_model)
        and _follows_lookup(registered_model, link_lookup)
    ]


def _follows_lookup(registered_model: type[Model], lookup: str) -> bool:
    """Return whether ``registered_model`` reads its rows' relations by ``lookup``."""
    return any(
        followed_lookup == lookup
        for followed_lookup, _ in rag.own_relation_lookups(registered_model)
    )


def _followed_lookups_named(
    registered_model: type[Model], lookup: str
) -> list[tuple[str, type[Model]]]:
    """Return ``registered_model``'s followed lookups named ``lookup``.

    Each comes with the model it reaches.
    """
    return [
        (followed_lookup, reached_model)
        for followed_lookup, reached_model in rag.relation_lookups(registered_model)
        if followed_lookup == lookup
    ]


def _link_relations(
    through: type[Model], instance: Model, model: type[Model], reverse: bool
) -> tuple[_LinkRelation, _LinkRelation] | None:
    """Return the relations the instance's side, then ``model``'s, reads links with.

    Those are the links of ``through``. ``reverse``, as m2m_changed sends it,
    is False when the instance's model declares the many-to-many: on a
    many-to-many from a model to itself, it tells the two sides apart. There are
    none when ``through`` is the through model of no many-to-many.
    """
    declaring_model = model if reverse else type(instance)
    # m2m_changed comes from a many-to-many of the declaring model, inherited
    # or not, and each many-to-many has a through model of its own.
    field = next(
        (
            field
            for field in declaring_model._meta.many_to_many
            if field.remote_field.through is through
        ),
        None,
    )
    if field is None:
        return None

    if field.remote_field.symmetrical:
        # A symmetrical many-to-many reads its links from both sides by the field.
        return field, field
    return (field.remote_field, field) if reverse else (field, field.remote_field)


def _registered_lookups_named(
    model: type[Model], link_lookup: str
) -> list[tuple[str, type[Model]]]:
    """Return the lookups by which a links change reaches rows following the links.

    ``link_lookup`` is the lookup ``model``'s side reads the changed links with.
    There are none when ``model`` is not registered or does not follow it.
    """
    if not rag.is_registered(model):
        return []
    return _followed_lookups_named(model, link_lookup)


def _followers_as_committed(
    sender: type[Model],
    instance: Model,
    followers_at_save: list[_Follower],
) -> list[_Follower]:
    """Return ``followers_at_save`` and the rows following ``instance`` now.

    On a create, ``followers_at_save`` are the followers the new row points
    to; on an update, only those it had before the save. The rows it points
    to now are looked up from the row as committed: on an update, those the
    save pointed it to, and those a write sending no signal has since pointed
    it to. Every row of a registered model reading ``sender`` by a lookup,
    following it through a to_field foreign key, or following a many-to-many
    whose through model it is, is replaced too, whichever row it reaches.
    If looking them up fails, ``followers_at_save`` are still replaced, but on
    an update a follower the row was moved to is not.
    """
    followed_source_key = _followed_source_key(sender, instance)
    try:
        return (
            _followers_pointed_to(sender, instance)
            + _every_row_reading(sender)
            + followers_at_save
        )
    except Exception:
        # An error escaping a commit callback would break the commit.
        logger.exception("Looking up the followers of %s failed", followed_source_key)
        return followers_at_save


def _every_row_reading(sender: type[Model]) -> list[_Follower]:
    """Return every row of the registered models a write to ``sender`` resyncs whole.

    Those models read ``sender`` by a lookup, follow it through a to_field
    foreign key, or follow a many-to-many whose through model it is: every row
    is returned, whichever row it reaches.
    """
    return _every_row_of(
        registered_model
        for registered_model in rag.registered_models()
        if _reads_besides_through(registered_model, sender)
        or _follows_through(registered_model, sender)
    )


def _every_row_reading_besides_through(sender: type[Model]) -> list[_Follower]:
    """Return every row of the registered models reading ``sender`` besides through.

    Those models read ``sender`` by a lookup or follow it through a to_field
    foreign key; following a many-to-many whose through model it is does not
    count: every row is returned, whichever row it reaches.
    """
    return _every_row_of(
        registered_model
        for registered_model in rag.registered_models()
        if _reads_besides_through(registered_model, sender)
    )


def _reads_besides_through(registered_model: type[Model], sender: type[Model]) -> bool:
    """Return whether ``registered_model`` reads ``sender`` by a lookup or to_field."""
    return bool(_followed_lookups(registered_model, sender)) or _follows_by_to_field(
        registered_model, sender
    )


def _every_row_of(registered_models: Iterable[type[Model]]) -> list[_Follower]:
    """Return every row of ``registered_models``, as followers."""
    return [
        (registered_model, follower_pk)
        for registered_model in registered_models
        for follower_pk in registered_model._base_manager.values_list("pk", flat=True)
    ]


def _is_through_of_followed(sender: type[Model]) -> bool:
    """Return whether a registered model follows a many-to-many through ``sender``."""
    return any(
        _follows_through(registered_model, sender)
        for registered_model in rag.registered_models()
    )


def _follows_through(registered_model: type[Model], sender: type[Model]) -> bool:
    """Return whether ``registered_model`` follows a many-to-many through ``sender``."""
    return any(
        concrete_model_of(through) is concrete_model_of(sender)
        for through in rag.through_models(registered_model)
    )


def _follows_by_to_field(registered_model: type[Model], sender: type[Model]) -> bool:
    """Return whether ``registered_model`` follows ``sender`` by a to_field key.

    A model registered not to resync on related writes follows none.
    """
    return rag.resyncs_on_related_writes(registered_model) and any(
        not relation.field.foreign_related_fields[0].primary_key
        for relation in _followed_reverse_relations(registered_model, sender)
    )


def _schedule_follower_replacements(
    followers: list[_Follower], followed_source_key: str
) -> None:
    """Replace, at the commit, the groups of ``followers``, one batch per model.

    A failure is logged with the ``followed_source_key`` of the row they follow.
    """
    if followers:
        _resync_followers_at_commit(followed_source_key, lambda: followers)


class _FollowerResync:
    """A commit callback replacing the groups of the followers a write changes.

    The callbacks of one transaction share the set of the followers already
    replaced, so a group followed by several writes is replaced once.
    """

    def __init__(
        self,
        followed_source_key: str,
        find_followers: Callable[[], list[_Follower]],
        replaced: set[_Follower],
    ) -> None:
        self.followed_source_key = followed_source_key
        self.find_followers = find_followers
        self.replaced = replaced
        self.has_run = False

    def __call__(self) -> None:
        self.has_run = True
        followers = [
            follower
            for follower in self.find_followers()
            if follower not in self.replaced
        ]
        self.replaced.update(followers)
        _replace_follower_groups(followers, self.followed_source_key)


def _resync_followers_at_commit(
    followed_source_key: str, find_followers: Callable[[], list[_Follower]]
) -> None:
    """Replace, at the commit, the groups ``find_followers`` returns then."""
    # The callbacks pending in the connection are those of the transaction.
    latest = next(
        (
            callback
            for _, callback, _ in reversed(transaction.get_connection().run_on_commit)
            if isinstance(callback, _FollowerResync)
        ),
        None,
    )
    # One that already ran belongs to an earlier commit run.
    replaced = set() if latest is None or latest.has_run else latest.replaced
    transaction.on_commit(
        _FollowerResync(followed_source_key, find_followers, replaced)
    )


def _replace_follower_groups(
    followers: list[_Follower], followed_source_key: str
) -> None:
    """Replace the groups of ``followers``, one batch per model."""
    pks_by_model: dict[type[Model], list[Any]] = {}
    # dict.fromkeys drops the duplicates and keeps the order.
    for follower_model, follower_pk in dict.fromkeys(followers):
        pks_by_model.setdefault(follower_model, []).append(follower_pk)
    for follower_model, follower_pks in pks_by_model.items():
        _replace_batch(follower_model, follower_pks, followed_source_key)


def _replace_batch(
    registered_model: type[Model], pks: list[Any], followed_source_key: str
) -> None:
    """Replace the groups of ``registered_model``'s rows ``pks``, as committed."""
    for start in range(0, len(pks), _PKS_PER_QUERY):
        chunk = pks[start : start + _PKS_PER_QUERY]
        _send_group(
            partial(_replace_groups, registered_model, chunk),
            f"{registered_model._meta.label_lower} instances that follow "
            f"{followed_source_key}",
        )


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

    They reach it through one foreign key of their own, a chain of foreign keys
    and one-to-ones, the one-to-ones forward or reverse, or a many-to-many.
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
            _lookups_to_group_pks(instance, followed_lookups(registered_model, sender)),
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

    Each crosses one foreign key, a chain of foreign keys and one-to-ones, the
    one-to-ones forward or reverse, or a many-to-many, and comes with the model
    it reaches, one that ``sender``'s rows are rows of.
    """
    followed_models = _models_of_the_row(sender)
    return [
        (lookup, reached_model)
        for lookup, reached_model in rag.relation_lookups(registered_model)
        # A foreign key may name a proxy: it reaches its concrete model's rows.
        if concrete_model_of(reached_model) in followed_models
    ]


def _lookups_to_group_pks(
    instance: Model, lookups: list[tuple[str, type[Model]]]
) -> list[tuple[str, Any]]:
    """Pair each of ``lookups`` with the primary key of the row it reaches.

    That row is ``instance``'s, as a row of the model the lookup comes with.
    """
    return [
        (lookup, _group_pk(instance, reached_model))
        for lookup, reached_model in lookups
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
    return (
        _is_followed_through_lookups(sender)
        or _is_followed_through_reverse_relations(sender)
        or _is_through_of_followed(sender)
    )


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

    Every row of a registered model reading ``sender`` by a lookup, following
    it through a to_field foreign key, or, unless the many-to-many's remove()
    or clear() deletes it, following a many-to-many whose through model it is,
    is replaced too, whichever row it reaches. Both happen once the
    transaction commits.
    """
    if not _signals_enabled():
        return

    # Django sends post_delete for each multi-table parent too, under its own
    # sender: the nearest registered model is the only group to empty here.
    nearest_registered_model_only = _registered_models(sender)[:1]
    # A many-to-many's remove() and clear() delete the through rows by a
    # queryset, after sending m2m_changed, which resyncs the models they change:
    # every other delete resyncs every row following it.
    every_row_reading = (
        _every_row_reading_besides_through
        if isinstance(kwargs.get("origin"), QuerySet)
        and _through_being_unlinked.get() is concrete_model_of(sender)
        else _every_row_reading
    )
    followers = (
        _reverse_followers(sender, instance)
        + instance.__dict__.pop(_REACHING_FOLLOWERS_ATTRIBUTE, [])
        + every_row_reading(sender)
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
