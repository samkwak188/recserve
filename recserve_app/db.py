import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import create_engine, select, delete, update, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from .schema_v1 import (users, identities, invitations, sessions, oauth_flows,
                        preferences, item_states, requests, events, mutations, outbox, rate_limits)


def now_ms():
    return time.time_ns() // 1_000_000


def hashed(value: str):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    user_id: str
    token_hash: str
    csrf_hash: str


class Database:
    def __init__(self, url):
        self.engine = create_engine(url, pool_size=4, max_overflow=4, pool_timeout=1,
            pool_pre_ping=True, connect_args={'connect_timeout': 3,
                'options': '-c statement_timeout=2000 -c lock_timeout=1000'})

    def rate(self, key, limit):
        window = now_ms() // 60000
        with self.engine.begin() as tx:
            stmt = pg_insert(rate_limits).values(key=key, window=window, count=1)
            count = tx.execute(stmt.on_conflict_do_update(
                index_elements=['key', 'window'], set_={'count': rate_limits.c.count + 1}
            ).returning(rate_limits.c.count)).scalar_one()
        if count > limit:
            raise HTTPException(429, 'Rate limit exceeded', headers={'Retry-After': '60'})

    def start_flow(self):
        state, browser, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(4))
        with self.engine.begin() as tx:
            tx.execute(insert(oauth_flows).values(state_hash=hashed(state), browser_hash=hashed(browser),
                       nonce=nonce, verifier=verifier, created_ms=now_ms()))
        return state, browser, nonce, verifier

    def consume_flow(self, state, browser):
        with self.engine.begin() as tx:
            row = tx.execute(delete(oauth_flows).where(
                oauth_flows.c.state_hash == hashed(state),
                oauth_flows.c.browser_hash == hashed(browser)).returning(oauth_flows)).mappings().first()
        if not row or now_ms() - row['created_ms'] > 600000:
            raise HTTPException(400, 'Invalid or expired login')
        return row

    def login(self, subject, email):
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        email = email.casefold()
        with self.engine.begin() as tx:
            invitation = tx.execute(select(invitations).where(invitations.c.email == email,
                invitations.c.enabled.is_(True)).with_for_update()).first()
            if not invitation:
                raise HTTPException(403, 'Invitation required')
            identity = tx.execute(select(identities).where(identities.c.subject == subject)).mappings().first()
            if identity:
                uid = identity['user_id']
                self.lock_user(tx, uid)
                tx.execute(update(identities).where(identities.c.subject == subject).values(email=email))
            else:
                uid = str(uuid.uuid4())
                tx.execute(insert(users).values(id=uid, created_ms=now_ms(), revision=0, disabled=False))
                tx.execute(insert(identities).values(subject=subject, user_id=uid, email=email))
            # Login rotates this account's sessions; bounded at one session for the pilot.
            tx.execute(delete(sessions).where(sessions.c.user_id == uid))
            tx.execute(insert(sessions).values(token_hash=hashed(token), user_id=uid,
                csrf_hash=hashed(csrf), created_ms=now_ms(), seen_ms=now_ms()))
        return token, csrf

    def authenticate(self, token):
        stamp = now_ms()
        with self.engine.begin() as tx:
            row = tx.execute(select(sessions).join(users, sessions.c.user_id == users.c.id).where(
                sessions.c.token_hash == hashed(token), users.c.disabled.is_(False),
                sessions.c.created_ms > stamp - 7 * 86400000,
                sessions.c.seen_ms > stamp - 86400000)).mappings().first()
            if not row:
                raise HTTPException(401, 'Authentication required')
            tx.execute(update(sessions).where(sessions.c.token_hash == row['token_hash']).values(seen_ms=stamp))
        return Principal(row['user_id'], row['token_hash'], row['csrf_hash'])

    @staticmethod
    def lock_user(tx, uid):
        row = tx.execute(select(users).where(users.c.id == uid).with_for_update()).mappings().first()
        if not row or row['disabled']:
            raise HTTPException(401, 'Authentication required')
        return row

    def delete_account(self, uid):
        with self.engine.begin() as tx:
            self.lock_user(tx, uid)
            email = tx.execute(select(identities.c.email).where(identities.c.user_id == uid)).scalar_one()
            tx.execute(delete(invitations).where(invitations.c.email == email))
            tx.execute(insert(outbox).values(id=str(uuid.uuid4()), user_id=uid, kind='delete', created_ms=now_ms()))
            tx.execute(delete(users).where(users.c.id == uid))

    def export(self, uid):
        result = {}
        with self.engine.begin() as tx:
            self.lock_user(tx, uid)
            for table in (identities, preferences, item_states, requests, events):
                result[table.name] = [dict(row) for row in tx.execute(select(table).where(table.c.user_id == uid)).mappings()]
        return result

    def maintenance(self):
        stamp = now_ms()
        with self.engine.begin() as tx:
            tx.execute(delete(oauth_flows).where(oauth_flows.c.created_ms < stamp - 600000))
            tx.execute(delete(sessions).where((sessions.c.created_ms < stamp - 7 * 86400000) |
                                            (sessions.c.seen_ms < stamp - 86400000)))
            for table in (events, requests, mutations):
                tx.execute(delete(table).where(table.c.created_ms < stamp - 90 * 86400000))
            tx.execute(delete(rate_limits).where(rate_limits.c.window < stamp // 60000 - 2))
