"""Frozen schema for migration 0001. Later migrations must not edit this snapshot."""
from sqlalchemy import (MetaData, Table, Column as C, String, BigInteger, Integer,
                        Boolean, ForeignKey, UniqueConstraint, CheckConstraint)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData()
users = Table('users', metadata,
    C('id', UUID(as_uuid=False), primary_key=True), C('created_ms', BigInteger, nullable=False),
    C('revision', BigInteger, nullable=False, default=0), C('consent_version', String(40)),
    C('consent_ms', BigInteger), C('disabled', Boolean, nullable=False, default=False))
identities = Table('identities', metadata,
    C('subject', String(255), primary_key=True),
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),
    C('email', String(320), nullable=False))
invitations = Table('invitations', metadata, C('email', String(320), primary_key=True),
    C('enabled', Boolean, nullable=False, default=True))
sessions = Table('sessions', metadata, C('token_hash', String(64), primary_key=True),
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
    C('csrf_hash', String(64), nullable=False), C('created_ms', BigInteger, nullable=False),
    C('seen_ms', BigInteger, nullable=False))
oauth_flows = Table('oauth_flows', metadata, C('state_hash', String(64), primary_key=True),
    C('browser_hash', String(64), nullable=False), C('nonce', String(128), nullable=False),
    C('verifier', String(128), nullable=False), C('created_ms', BigInteger, nullable=False))
preferences = Table('preferences', metadata,
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    C('item_id', Integer, primary_key=True), C('value', Integer, nullable=False),
    CheckConstraint('value IN (-1, 1)', name='preference_value'))
item_states = Table('item_states', metadata,
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    C('item_id', Integer, primary_key=True),
    C('saved', Boolean, nullable=False, default=False), C('watched', Boolean, nullable=False, default=False),
    C('dismissed', Boolean, nullable=False, default=False))
requests = Table('recommendations', metadata,
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    C('request_id', UUID(as_uuid=False), primary_key=True), C('k', Integer, nullable=False),
    C('created_ms', BigInteger, nullable=False), C('response', JSONB, nullable=False))
events = Table('events', metadata,
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    C('event_id', UUID(as_uuid=False), primary_key=True), C('request_id', UUID(as_uuid=False), nullable=False),
    C('item_id', Integer, nullable=False), C('kind', String(16), nullable=False),
    C('created_ms', BigInteger, nullable=False), C('payload', JSONB, nullable=False),
    UniqueConstraint('user_id', 'request_id', 'item_id', 'kind', name='one_action_per_impression'))
mutations = Table('mutations', metadata,
    C('user_id', UUID(as_uuid=False), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    C('request_id', UUID(as_uuid=False), primary_key=True), C('payload', JSONB, nullable=False),
    C('response', JSONB, nullable=False), C('created_ms', BigInteger, nullable=False))
models = Table('models', metadata, C('digest', String(64), primary_key=True),
    C('metadata', JSONB, nullable=False), C('created_ms', BigInteger, nullable=False))
outbox = Table('privacy_outbox', metadata, C('id', UUID(as_uuid=False), primary_key=True),
    C('user_id', UUID(as_uuid=False), nullable=False), C('kind', String(16), nullable=False),
    C('created_ms', BigInteger, nullable=False), C('delivered_ms', BigInteger))
rate_limits = Table('rate_limits', metadata, C('key', String(128), primary_key=True),
    C('window', BigInteger, primary_key=True), C('count', Integer, nullable=False))
