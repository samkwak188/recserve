import time
from fastapi import HTTPException
from sqlalchemy import select, insert, update, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .db import now_ms
from .schema_v1 import users, preferences, item_states, requests, events, mutations, models
from .selection import select_items


class Policy:
    def __init__(self, db, model, retrieval, consent_version):
        self.db, self.model, self.retrieval, self.consent_version = db, model, retrieval, consent_version
        with db.engine.begin() as tx:
            tx.execute(pg_insert(models).values(digest=model.digest, metadata=model.manifest,
                       created_ms=now_ms()).on_conflict_do_nothing(index_elements=['digest']))

    def account(self, tx, uid):
        user = self.db.lock_user(tx, uid)
        if user['consent_version'] != self.consent_version:
            raise HTTPException(403, 'Research consent required')
        return user

    def change_preferences(self, uid, payload):
        body = payload.model_dump()
        with self.db.engine.begin() as tx:
            user = self.account(tx, uid)
            old = tx.execute(select(mutations).where(mutations.c.user_id == uid,
                mutations.c.request_id == payload.request_id)).mappings().first()
            if old:
                if old['payload'] != body:
                    raise HTTPException(409, 'Idempotency key conflict')
                return old['response']
            if len({c.item_id for c in payload.changes}) != len(payload.changes):
                raise HTTPException(422, 'Duplicate preference item')
            for change in payload.changes:
                if change.item_id not in self.model.rows:
                    raise HTTPException(422, 'Unknown movie')
                if change.value == 0:
                    tx.execute(delete(preferences).where(preferences.c.user_id == uid, preferences.c.item_id == change.item_id))
                else:
                    stmt = pg_insert(preferences).values(user_id=uid, item_id=change.item_id, value=change.value)
                    tx.execute(stmt.on_conflict_do_update(index_elements=['user_id', 'item_id'], set_={'value': change.value}))
            count = len(tx.execute(select(preferences.c.item_id).where(preferences.c.user_id == uid)).all())
            if count > 500:
                raise HTTPException(422, 'Preference limit reached')
            revision = user['revision'] + 1
            tx.execute(update(users).where(users.c.id == uid).values(revision=revision))
            response = {'preference_revision': revision, 'count': count}
            tx.execute(insert(mutations).values(user_id=uid, request_id=payload.request_id, payload=body,
                                              response=response, created_ms=now_ms()))
            return response

    def recommend(self, uid, payload):
        self.db.rate('recommend:' + uid, 20)
        deadline = time.monotonic() + .3
        for attempt in range(3):
            with self.db.engine.begin() as tx:
                user = self.account(tx, uid)
                old = tx.execute(select(requests).where(requests.c.user_id == uid,
                    requests.c.request_id == payload.request_id)).mappings().first()
                if old:
                    if old['k'] != payload.k or now_ms() - old['created_ms'] >= 86400000:
                        raise HTTPException(409, 'Request key conflicts or has expired; use a new key')
                    return old['response']
                prefs = dict(tx.execute(select(preferences.c.item_id, preferences.c.value).where(preferences.c.user_id == uid)).all())
                blocked = set(prefs) | set(tx.execute(select(item_states.c.item_id).where(item_states.c.user_id == uid,
                    item_states.c.saved | item_states.c.watched | item_states.c.dismissed)).scalars())
                revision = user['revision']
            vector = self.model.vector(uid, revision, prefs)
            selection = select_items(self.model, self.retrieval, vector, blocked, payload.k,
                                     self.model.policy, deadline)
            response = dict(request_id=payload.request_id, model_version=self.model.digest,
                policy_version='explicit-v2-' + self.model.policy, preference_revision=revision,
                **selection)
            with self.db.engine.begin() as tx:
                current = self.account(tx, uid)
                old = tx.execute(select(requests).where(requests.c.user_id == uid,
                    requests.c.request_id == payload.request_id)).mappings().first()
                if old:
                    if old['k'] != payload.k:
                        raise HTTPException(409, 'Request key conflict')
                    return old['response']
                if current['revision'] != revision:
                    continue
                tx.execute(insert(requests).values(user_id=uid, request_id=payload.request_id,
                           k=payload.k, created_ms=now_ms(), response=response))
                return response
        raise HTTPException(503, 'Preferences changed repeatedly; retry with a new request ID')

    def event(self, uid, payload):
        body = payload.model_dump()
        with self.db.engine.begin() as tx:
            user = self.account(tx, uid)
            old = tx.execute(select(events).where(events.c.user_id == uid, events.c.event_id == payload.event_id)).mappings().first()
            if old:
                if old['payload'] != body:
                    raise HTTPException(409, 'Event key conflict')
                return {'accepted': True, 'duplicate': True}
            recommendation = tx.execute(select(requests).where(requests.c.user_id == uid,
                requests.c.request_id == payload.request_id)).mappings().first()
            if not recommendation or payload.item_id not in {item['id'] for item in recommendation['response']['items']}:
                raise HTTPException(422, 'Event must reference an issued movie')
            stamp = now_ms()
            if not max(stamp - 86400000, recommendation['created_ms'] - 30000) <= payload.event_time_ms <= stamp + 30000:
                raise HTTPException(422, 'Event timestamp out of range')
            previous = set(tx.execute(select(events.c.kind).where(events.c.user_id == uid,
                events.c.request_id == payload.request_id, events.c.item_id == payload.item_id)).scalars())
            if payload.kind in previous:
                raise HTTPException(409, 'Action already recorded with another event ID')
            if payload.kind != 'shown' and 'shown' not in previous:
                raise HTTPException(409, 'Acknowledge display before an outcome')
            tx.execute(insert(events).values(user_id=uid, event_id=payload.event_id, request_id=payload.request_id,
                item_id=payload.item_id, kind=payload.kind, created_ms=stamp, payload=body))
            if payload.kind != 'shown':
                column = {'save': 'saved', 'dismiss': 'dismissed', 'watched': 'watched'}[payload.kind]
                stmt = pg_insert(item_states).values(user_id=uid, item_id=payload.item_id, **{column: True})
                tx.execute(stmt.on_conflict_do_update(index_elements=['user_id', 'item_id'], set_={column: True}))
                tx.execute(update(users).where(users.c.id == uid).values(revision=user['revision'] + 1))
            return {'accepted': True, 'duplicate': False}
