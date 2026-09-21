"""Single-host transactional feedback journal and projection.

No broker offsets or asynchronous snapshot publisher: the database is both the
journal and serving projection. A generation fences an in-flight response when
feedback changes while candidate retrieval is outside the transaction.
"""
from contextlib import contextmanager
import json
import sqlite3
import time


def now_ms():
    return time.time_ns() // 1_000_000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Conflict(ValueError):
    pass


class Changed(RuntimeError):
    pass


def identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(ord(c) < 33 for c in value):
        raise ValueError('IDs must be nonempty strings of at most 128 non-whitespace characters')
    return value


class Store:
    def __init__(self, path, fingerprint):
        self.path = str(path)
        with self.connection() as db:
            # WAL is for a local filesystem only. FULL requests commit durability;
            # process-crash tests do not certify hardware power-loss behavior.
            mode = db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
            if mode != 'wal':
                raise RuntimeError('pilot requires a file-backed WAL database')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS meta (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    fingerprint TEXT NOT NULL, schema_version INTEGER NOT NULL,
                    generation INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS impressions (
                    request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    k INTEGER NOT NULL, created_ms INTEGER NOT NULL, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS exposures (
                    request_id TEXT NOT NULL REFERENCES impressions(request_id),
                    item_id TEXT NOT NULL, PRIMARY KEY(request_id,item_id));
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
                    request_id TEXT NOT NULL REFERENCES impressions(request_id),
                    item_id TEXT NOT NULL, kind TEXT NOT NULL,
                    committed_ms INTEGER NOT NULL,
                    UNIQUE(request_id,item_id,kind));
                CREATE TABLE IF NOT EXISTS blocked (
                    user_id TEXT NOT NULL, item_id TEXT NOT NULL,
                    PRIMARY KEY(user_id,item_id));
                CREATE TABLE IF NOT EXISTS item_features (
                    item_id TEXT PRIMARY KEY, shown INTEGER NOT NULL DEFAULT 0,
                    saved INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS unavailable (item_id TEXT PRIMARY KEY);
            ''')
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO meta VALUES (1,?,1,0)', (fingerprint,))
            row = db.execute('SELECT fingerprint,schema_version FROM meta').fetchone()
            if row != (fingerprint, 1):
                raise ValueError('database model/schema mismatch; use an explicit migration or separate database')
            db.commit()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=0.25, isolation_level=None)
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA synchronous=FULL')
            yield db
        finally:
            if db.in_transaction:
                db.rollback()
            db.close()

    @staticmethod
    def cached(db, request_id, user, k):
        row = db.execute('SELECT user_id,k,response FROM impressions WHERE request_id=?', (request_id,)).fetchone()
        if row is None:
            return None
        if row[:2] != (user, k):
            raise Conflict('request ID already used with different inputs')
        return json.loads(row[2])

    def existing(self, request_id, user, k):
        with self.connection() as db:
            return self.cached(db, request_id, user, k)

    def snapshot(self, user):
        with self.connection() as db:
            db.execute('BEGIN')
            generation = db.execute('SELECT generation FROM meta').fetchone()[0]
            blocked = {r[0] for r in db.execute('SELECT item_id FROM blocked WHERE user_id=?', (user,))}
            unavailable = {r[0] for r in db.execute('SELECT item_id FROM unavailable')}
            features = {r[0]: (r[1], r[2]) for r in db.execute('SELECT item_id,shown,saved FROM item_features')}
            last = db.execute('SELECT MAX(committed_ms) FROM events').fetchone()[0]
            return dict(generation=generation, blocked=blocked, unavailable=unavailable,
                        features=features, last_committed_ms=last)

    def publish(self, response, k):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            cached = self.cached(db, response['request_id'], response['user_id'], k)
            if cached is not None:
                return cached
            if db.execute('SELECT generation FROM meta').fetchone()[0] != response['feature_generation']:
                raise Changed('feedback changed during retrieval')
            db.execute('INSERT INTO impressions VALUES (?,?,?,?,?)',
                       (response['request_id'], response['user_id'], k, now_ms(), canonical(response)))
            db.executemany('INSERT INTO exposures VALUES (?,?)',
                           [(response['request_id'], item['item_id']) for item in response['items']])
            db.commit()
            return response

    def apply(self, event, crash=None):
        keys = {'schema_version', 'event_id', 'event_time_ms', 'user_id', 'item_id', 'kind', 'request_id'}
        if not isinstance(event, dict) or set(event) != keys:
            raise ValueError('event schema fields do not match version 1')
        if type(event['schema_version']) is not int or event['schema_version'] != 1:
            raise ValueError('unsupported event schema')
        for key in ('event_id', 'user_id', 'item_id', 'request_id'):
            identifier(event[key])
        if event['kind'] not in ('shown', 'save', 'dismiss', 'watched'):
            raise ValueError('unsupported feedback kind')
        if type(event['event_time_ms']) is not int:
            raise ValueError('event time must be epoch milliseconds')
        payload = canonical(event)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT sequence,payload FROM events WHERE event_id=?', (event['event_id'],)).fetchone()
            if old:
                if old[1] != payload:
                    raise Conflict('event ID reused with different payload')
                return dict(accepted=True, duplicate=True, sequence=old[0])
            stamp = now_ms()
            if not stamp - 86_400_000 <= event['event_time_ms'] <= stamp + 30_000:
                raise ValueError('event older than 24 hours or more than 30 seconds in the future')
            impression = db.execute('SELECT user_id,created_ms FROM impressions WHERE request_id=?',
                                    (event['request_id'],)).fetchone()
            exposed = db.execute('SELECT 1 FROM exposures WHERE request_id=? AND item_id=?',
                                 (event['request_id'], event['item_id'])).fetchone()
            if not impression or impression[0] != event['user_id'] or not exposed:
                raise ValueError('feedback must reference an issued item for the same user')
            if event['event_time_ms'] < impression[1] - 30_000:
                raise ValueError('feedback precedes its recommendation')
            if event['kind'] != 'shown' and not db.execute(
                    "SELECT 1 FROM events WHERE request_id=? AND item_id=? AND kind='shown'",
                    (event['request_id'], event['item_id'])).fetchone():
                raise Conflict('acknowledge shown before recording an outcome')
            try:
                cursor = db.execute('INSERT INTO events(event_id,payload,request_id,item_id,kind,committed_ms) VALUES (?,?,?,?,?,?)',
                                    (event['event_id'], payload, event['request_id'], event['item_id'], event['kind'], stamp))
            except sqlite3.IntegrityError as exc:
                raise Conflict('same impression/item action already recorded under another event ID') from exc
            sequence = cursor.lastrowid
            db.execute('INSERT OR IGNORE INTO item_features(item_id) VALUES (?)', (event['item_id'],))
            if event['kind'] == 'shown':
                db.execute('UPDATE item_features SET shown=shown+1 WHERE item_id=?', (event['item_id'],))
            else:
                db.execute('INSERT OR IGNORE INTO blocked VALUES (?,?)', (event['user_id'], event['item_id']))
                if event['kind'] == 'save':
                    db.execute('UPDATE item_features SET saved=saved+1 WHERE item_id=?', (event['item_id'],))
            db.execute('UPDATE meta SET generation=generation+1')
            if crash:
                crash('before_commit')
            db.commit()
            if crash:
                crash('after_commit')
            return dict(accepted=True, duplicate=False, sequence=sequence)

    def set_unavailable(self, item, value):
        """Local operator hook, not an unauthenticated HTTP mutation endpoint."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if value:
                db.execute('INSERT OR IGNORE INTO unavailable VALUES (?)', (identifier(item),))
            else:
                db.execute('DELETE FROM unavailable WHERE item_id=?', (identifier(item),))
            db.execute('UPDATE meta SET generation=generation+1')
            db.commit()
