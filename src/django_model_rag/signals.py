"""Signal receivers that keep the output in step with saved and deleted instances."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any, TypeAlias, TypeGuard, cast

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import (
    ForeignObject,
    ForeignObjectRel,
    ManyToManyField,
    ManyToManyRel,
    Model,
    QuerySet,
)

from django_model_rag.documents import model_source_key
from django_model_rag.extractors import through_key_to_parent
from django_model_rag.output import check_output_configuration, configured_output
from django_model_rag.pipeline import SyncPipeline
from django_model_rag.registry import concrete_model_of, rag

_SIGNALS_SETTING = "MODEL_RAG_SIGNALS"
_CHANGING_ACTIONS = frozenset({"post_add", "post_remove", "post_clear"})
_BEFORE_CLEAR = "pre_clear"
_BEFORE_WRITE = frozenset({"pre_add", "pre_remove", _BEFORE_CLEAR})
# The instance carries the keys of the rows its links reach, from before a clear.
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

    # Followers reaching the row through foreign keys are looked up by its
    # primary key, joined against the columns still in the database: pre_save
    # runs before the UPDATE, so they are those naming the row as committed,
    # even by columns the save changes, without loading it. At the commit, both
    # these and the reverse followers are looked up from the row as saved: only
    # the lookups made here find the followers it had before the save.
    setattr(
        instance,
        _PREVIOUS_FOLLOWERS_ATTRIBUTE,
        _committed_reverse_followers(sender, instance.pk)
        + _followers_reaching(sender, instance),
    )


def _committed_reverse_followers(sender: type[Model], pk: Any) -> list[_Follower]:
    """Return the rows following, through a reverse relation, the committed row.

    The row points to them by its foreign keys, read as committed: before the
    save, they are those it pointed to before, even if the save changes them;
    at the commit, those it points to then, even through a write sending no
    signal.
    """
    # A model followed only through foreign keys costs no row loading.
    if not _is_followed_through_reverse_relations(sender):
        return []

    if pk is None:
        # A row deleted since its save has lost its primary key: no committed
        # row has it, and looking for one would query for nothing.
        return []

    committed_instance = _committed_instance(sender._base_manager.all(), pk)
    if committed_instance is None:
        return []

    return _reverse_followers(sender, committed_instance)


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
    The followers of a row saved before are looked up again at the commit. The
    output configuration was checked before the save, by
    check_output_before_save.
    """
    if raw or not _signals_enabled():
        return

    registered_models = _registered_models(sender)
    _schedule_commit_callbacks(registered_models, instance, _group_replacer)
    # At post_save the instance is the row as saved: a change left unsaved in
    # memory afterwards is not followed.
    followers_at_save = _reverse_followers(sender, instance) + instance.__dict__.pop(
        _PREVIOUS_FOLLOWERS_ATTRIBUTE, []
    )
    # A row just created has no follower pointing to it at its save.
    if not created and _is_followed(sender):
        # Rows may be attached to it, or it to other rows, before the commit,
        # by a write that sends no signal: its followers are looked up then.
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
    model: type[Model] | None = None,
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
    if action in _BEFORE_WRITE:
        # Keys left by a clear that failed after pre_clear are not this change's.
        instance.__dict__.pop(_CLEARED_PKS_ATTRIBUTE, None)
        _check_output_before_write(through, instance, model)
        if action == _BEFORE_CLEAR:
            _remember_cleared_pks(through, instance, model, reverse)
        return

    if action not in _CHANGING_ACTIONS:
        return

    _schedule_commit_callbacks(
        _registered_models_following(type(instance), through),
        instance,
        _group_replacer,
    )
    if _reaches_registered_rows(model, through):
        # Only a clear leaves keys behind, found before it: pk_set is None then.
        pk_set = (pk_set or set()) | instance.__dict__.pop(
            _CLEARED_PKS_ATTRIBUTE, set()
        )
        _, key_to_model = _through_keys(through, instance, model, reverse)
        _schedule_follower_replacements(
            [(model, pk) for pk in _primary_keys_named(key_to_model, model, pk_set)],
            _followed_source_key(type(instance), instance),
        )


def _check_output_before_write(
    through: type[Model], instance: Model, model: type[Model] | None
) -> None:
    """Fail before the join rows are written or deleted if the sync would fail.

    In autocommit the join rows are committed as soon as they are written:
    after the change, a failing check would come too late to undo it.
    """
    if _registered_models_following(type(instance), through) or (
        _reaches_registered_rows(model, through)
    ):
        check_output_configuration()


def _registered_models_following(
    sender: type[Model], through: type[Model]
) -> list[type[Model]]:
    """Return the registered models of ``sender`` following the links of ``through``."""
    return [
        registered_model
        for registered_model in _registered_models(sender)
        if rag.follows_many_to_many(registered_model, through)
    ]


def _through_keys(
    through: type[Model], instance: Model, model: type[Model], reverse: bool
) -> tuple["ForeignObject[Any, Any]", "ForeignObject[Any, Any]"]:
    """Return the foreign keys of ``through`` to the instance's side, then ``model``'s.

    ``reverse``, as m2m_changed sends it, is False when the instance's model
    declares the many-to-many.
    """
    declaring_model = model if reverse else type(instance)
    # m2m_changed comes from a many-to-many of the declaring model, inherited
    # or not, and each many-to-many has a through model of its own.
    field = next(
        field
        for field in declaring_model._meta.many_to_many
        if field.remote_field.through is through
    )
    instance_side, model_side = (
        (field.remote_field, field) if reverse else (field, field.remote_field)
    )
    return _through_key(instance_side), _through_key(model_side)


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


def _reaches_registered_rows(
    model: type[Model] | None, through: type[Model]
) -> TypeGuard[type[Model]]:
    """Return whether a links change reaches rows following the links."""
    return (
        model is not None
        and rag.is_registered(model)
        and rag.follows_many_to_many(model, through)
    )


def _remember_cleared_pks(
    through: type[Model], instance: Model, model: type[Model] | None, reverse: bool
) -> None:
    """Keep the keys of the registered rows a clear is about to unlink.

    Django sends no primary keys with the clear: they can only be found before
    it. ``reverse`` is m2m_changed's.
    """
    if not _reaches_registered_rows(model, through):
        return

    key_to_instance, key_to_model = _through_keys(through, instance, model, reverse)
    # The foreign key may name the instance by a unique column other than its
    # primary key, and a multi-table child by the row of the parent holding the
    # links.
    links = through._base_manager.filter(
        **{
            key_to_instance.name: getattr(
                instance, key_to_instance.foreign_related_fields[0].attname
            )
        }
    )
    setattr(
        instance,
        _CLEARED_PKS_ATTRIBUTE,
        set(links.values_list(key_to_model.attname, flat=True)),
    )


def _replace_followers_as_committed(
    sender: type[Model],
    instance: Model,
    followers_at_save: list[_Follower],
) -> None:
    """Replace the groups of ``followers_at_save`` and of the rows following it now.

    Those are looked up from the row as committed: rows attached after the
    save, in the same transaction, follow it too, and so do the rows a write
    sending no signal has since pointed it to. If looking them up fails,
    ``followers_at_save`` are still replaced.
    """
    followed_source_key = _followed_source_key(sender, instance)
    try:
        followers = (
            _followers_reaching(sender, instance)
            + _committed_reverse_followers(sender, instance.pk)
            + followers_at_save
        )
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
    return [
        (registered_model, follower_pk)
        for registered_model in rag.registered_models()
        for lookup, reached_model in _followed_lookups(registered_model, sender)
        for follower_pk in _pks_reaching(
            registered_model, lookup, _group_pk(instance, reached_model)
        )
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
    registered_model: type[Model], lookup: str, reached_pk: Any
) -> list[Any]:
    """Return the primary keys of the ``registered_model`` rows reaching a row.

    Those rows reach, through ``lookup``, the row whose primary key is ``reached_pk``.
    """
    if reached_pk is None:
        # A row deleted since its save has lost its primary key: filtering on
        # None would match the rows whose foreign key is null.
        return []

    reaching_rows = registered_model._base_manager.filter(
        **{f"{lookup}__pk": reached_pk}
    )
    return list(reaching_rows.values_list("pk", flat=True))


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
