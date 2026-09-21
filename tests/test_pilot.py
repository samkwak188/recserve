"""Policy, transactional races and actual process-crash recovery; no network data."""
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import subprocess
import socket
import threading
import time
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import closing
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recserve_pilot.model import eligible, rank
from recserve_pilot.service import Pilot, Server
from recserve_pilot.store import Changed, Conflict, Store, now_ms


class Candidates:
    def __init__(self):
        self.calls = []

    def retrieve(self, user, count):
        self.calls.append(count)
        return {str(i): 1-i/200 for i in range(count)}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'state.sqlite'
        self.store = Store(self.path, 'fixture-sha')
        items = [str(i) for i in range(150)]
        self.model = SimpleNamespace(users={'u': 0}, items=items, item_rows={x: int(x) for x in items},
                                     seen={'u': {'0'}}, popular=items, version='test', fingerprint='fixture-sha')
        self.upstream = Candidates()
        self.pilot = Pilot(self.model, self.store, self.upstream)

    def recommend(self, rid='r1', user='u', k=5):
        return self.pilot.recommend(dict(request_id=rid, user_id=user, k=k))

    def event(self, kind='shown', event_id='e1', request_id='r1', item=None, user='u'):
        if item is None:
            item = self.recommend()['items'][0]['item_id']
        return dict(schema_version=1, event_id=event_id, event_time_ms=now_ms(),
                    user_id=user, item_id=item, kind=kind, request_id=request_id)

    def test_dismiss_changes_response_and_survives_restart(self):
        before = self.recommend()
        shown = self.event()
        self.store.apply(shown)
        event = self.event('dismiss', 'e2', item=shown['item_id'])
        self.store.apply(event)
        self.pilot.store = Store(self.path, 'fixture-sha')
        after = self.recommend('r2')
        self.assertNotIn(shown['item_id'], [x['item_id'] for x in after['items']])
        self.assertEqual(after['feature_generation'], 2)
        self.assertNotEqual(before['items'], after['items'])

    def test_event_duplicate_and_conflicting_payload(self):
        event = self.event()
        first = self.store.apply(event)
        duplicate = self.store.apply(event)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(first['sequence'], duplicate['sequence'])
        self.assertEqual(self.store.snapshot('u')['generation'], 1)
        with self.assertRaises(Conflict):
            self.store.apply(dict(event, kind='dismiss'))
        with self.assertRaises(Conflict):
            self.store.apply(dict(event, event_id='another-id'))

    def test_recommendation_idempotency(self):
        first = self.recommend()
        self.store.apply(self.event())
        self.assertEqual(first, self.recommend())
        with self.assertRaises(Conflict):
            self.recommend(user='guest:someone')

    def test_ack_required_and_attribution(self):
        with self.assertRaises(Conflict):
            self.store.apply(self.event('save'))
        for field, value in [('user_id', 'other'), ('item_id', '149'), ('request_id', 'missing')]:
            with self.assertRaises(ValueError):
                self.store.apply(dict(self.event(), **{field: value}))

    def test_time_schema_and_limits(self):
        event = self.event()
        for modification in [dict(event_time_ms=True), dict(event_time_ms=now_ms()+60_000),
                             dict(event_time_ms=now_ms()-86_500_000), dict(schema_version=True),
                             dict(kind='click'), dict(event_id=''), dict(extra=1)]:
            with self.assertRaises(ValueError):
                self.store.apply(dict(event, **modification))
        for k in [0, 51, True, '5']:
            with self.assertRaises(ValueError):
                self.recommend(k=k)

    def test_concurrent_duplicate_is_one_transaction(self):
        event = self.event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(self.store.apply, [event]*16))
        self.assertEqual(sum(not r['duplicate'] for r in results), 1)
        self.assertEqual(self.store.snapshot('u')['features'][event['item_id']], (1, 0))

    def test_process_crashes_before_and_after_commit(self):
        event = self.event()
        event_file = Path(self.temp.name)/'event.json'
        event_file.write_text(json.dumps(event), encoding='utf-8')
        worker = ROOT/'tests/pilot_crash_worker.py'
        for boundary, generation in [('before_commit', 0), ('after_commit', 1)]:
            result = subprocess.run([sys.executable, str(worker), str(self.path), str(event_file), boundary],
                                    cwd=ROOT, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 77, result.stderr.decode())
            restored = Store(self.path, 'fixture-sha')
            self.assertEqual(restored.snapshot('u')['generation'], generation)
        self.assertTrue(self.store.apply(event)['duplicate'])
        self.assertEqual(self.store.snapshot('u')['features'][event['item_id']], (1, 0))

    def test_generation_fence_rejects_stale_publication(self):
        response = self.recommend()
        self.store.apply(self.event())
        with self.assertRaises(Changed):
            self.store.publish(dict(response, request_id='stale'), 5)

    def test_feedback_during_retrieval_retries(self):
        event = self.event()
        self.store.apply(event)
        original = self.upstream.retrieve
        calls = []
        def racing(user, count):
            if not calls:
                calls.append(1)
                self.store.apply(dict(event, event_id='dismiss-race', kind='dismiss'))
            return original(user, count)
        self.upstream.retrieve = racing
        response = self.recommend('racing')
        self.assertNotIn(event['item_id'], [x['item_id'] for x in response['items']])
        self.assertEqual(response['feature_generation'], 2)

    def test_replenishment_and_training_seen(self):
        self.model.seen['u'] = {str(i) for i in range(70)}
        response = self.recommend()
        self.assertEqual(self.upstream.calls, [64, 128])
        self.assertEqual(response['items'][0]['item_id'], '70')

    def test_all_blocked_is_explicit_exhaustion(self):
        self.model.seen['u'] = set(self.model.items)
        response = self.recommend()
        self.assertEqual(response['items'], [])
        self.assertTrue(response['exhausted'])

    def test_cold_start_and_upstream_failure(self):
        self.assertEqual(self.recommend(user='guest:new')['candidate_source'], 'popularity_cold_start')
        self.assertEqual(self.upstream.calls, [])
        def failed(*_args):
            raise OSError('device unavailable')
        self.upstream.retrieve = failed
        response = self.recommend('offline')
        self.assertTrue(response['degraded'])
        self.assertNotIn('0', [x['item_id'] for x in response['items']])
        with self.assertRaises(ValueError):
            self.recommend('unknown', user='unknown')

    def test_save_changes_item_feature_and_excludes_saved(self):
        event = self.event()
        self.store.apply(event)
        before = self.store.snapshot('u')
        self.store.apply(dict(event, kind='save', event_id='save'))
        after = self.store.snapshot('u')
        self.assertEqual(after['features'][event['item_id']], (1, 1))
        candidate = {event['item_id']: 0.5}
        before['blocked'], after['blocked'] = set(), set()
        self.assertGreater(rank(candidate, set(), after, self.model.item_rows)[0][1],
                           rank(candidate, set(), before, self.model.item_rows)[0][1])
        self.assertNotIn(event['item_id'], [x['item_id'] for x in self.recommend('next')['items']])

    def test_availability_and_eligibility_parity(self):
        self.store.set_unavailable('1', True)
        response = self.recommend()
        snapshot = self.store.snapshot('u')
        for item in response['items']:
            self.assertTrue(eligible(item['item_id'], {'0'}, snapshot['blocked'], snapshot['unavailable']))
        self.assertNotIn('1', [x['item_id'] for x in response['items']])
        self.store.set_unavailable('1', False)
        self.assertEqual(self.recommend('available')['items'][0]['item_id'], '1')

    def test_projection_matches_independent_journal_replay(self):
        response = self.recommend()
        for index, item in enumerate(response['items']):
            event = self.event(event_id=f'shown-{index}', item=item['item_id'])
            self.store.apply(event)
            self.store.apply(dict(event, event_id=f'outcome-{index}', kind='save' if index%2 else 'dismiss'))
        shown, saved, blocked = Counter(), Counter(), set()
        with self.store.connection() as db:
            events = [json.loads(r[0]) for r in db.execute('SELECT payload FROM events ORDER BY sequence')]
        for event in events:
            item = event['item_id']
            if event['kind'] == 'shown':
                shown[item] += 1
            else:
                blocked.add(item)
                saved[item] += event['kind'] == 'save'
        state = self.store.snapshot('u')
        self.assertEqual(state['blocked'], blocked)
        self.assertEqual(state['features'], {item: (count, saved[item]) for item, count in shown.items()})
        self.assertEqual(state['generation'], len(events))

    def test_backup_restore_and_model_mismatch(self):
        self.store.apply(self.event())
        backup_path = Path(self.temp.name)/'backup.sqlite'
        with self.store.connection() as source, closing(sqlite3.connect(backup_path)) as dest:
            source.backup(dest)
        restored = Store(backup_path, 'fixture-sha')
        self.assertEqual(restored.snapshot('u'), self.store.snapshot('u'))
        with self.assertRaises(ValueError):
            Store(self.path, 'different-model')

    def test_http_slow_stream_has_absolute_lifetime(self):
        server = Server(('127.0.0.1', 0), self.pilot)
        server.request_lifetime_s = 0.2
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        try:
            with socket.create_connection(server.server_address, timeout=2) as client:
                start = time.monotonic()
                client.sendall(b'POST /v1/events HTTP/1.1\r\nX-Slow: ')
                for _ in range(20):
                    try:
                        client.sendall(b'a')
                    except OSError:
                        break
                    time.sleep(.03)
                try:
                    self.assertEqual(client.recv(1), b'')
                except (ConnectionAbortedError, ConnectionResetError):
                    pass  # Windows can abort rather than deliver EOF on shutdown.
                self.assertLess(time.monotonic()-start, 1)
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()

    def test_http_admission_is_bounded_and_recovers(self):
        server = Server(('127.0.0.1', 0), self.pilot)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        clients = []
        try:
            for _ in range(8):
                client = socket.create_connection(server.server_address, timeout=2)
                client.sendall(b'G')
                clients.append(client)
            until = time.monotonic()+2
            while server.slots._value != 0 and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(server.slots._value, 0)
            with socket.create_connection(server.server_address, timeout=2) as excess:
                self.assertEqual(excess.recv(1), b'')
            for client in clients:
                client.close()
            clients.clear()
            until = time.monotonic()+2
            while server.slots._value != 8 and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(server.slots._value, 8)
        finally:
            for client in clients:
                client.close()
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()


if __name__ == '__main__':
    unittest.main()
