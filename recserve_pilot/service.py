"""Local policy layer with generation-fenced feedback and bounded admission."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time

from .model import Model, rank
from .store import Changed, Conflict, Store, identifier
from .transport import Upstream


class Pilot:
    def __init__(self, model, store, upstream):
        self.model, self.store, self.upstream = model, store, upstream

    def recommend(self, payload):
        if not isinstance(payload, dict) or set(payload) != {'request_id', 'user_id', 'k'}:
            raise ValueError('recommend expects request_id, user_id and k')
        request_id, user, k = identifier(payload['request_id']), identifier(payload['user_id']), payload['k']
        if type(k) is not int or not 1 <= k <= 50:
            raise ValueError('k must be an integer in [1,50]')
        if user not in self.model.users and not user.startswith('guest:'):
            raise ValueError('unknown model identity; use a distinct guest: ID for cold start')
        old = self.store.existing(request_id, user, k)
        if old is not None:
            return old
        for _ in range(3):
            snapshot = self.store.snapshot(user)
            candidates = {}
            source = 'popularity_cold_start'
            seen = self.model.seen.get(user, set())
            ranked = []
            if user in self.model.users:
                count = min(64, len(self.model.items))
                while True:
                    try:
                        candidates.update(self.upstream.retrieve(self.model.users[user], count))
                        source = 'cpp_retrieval'
                    except (OSError, TimeoutError):
                        candidates.clear()
                        source = 'popularity_upstream_unavailable'
                        break
                    ranked = rank(candidates, seen, snapshot, self.model.item_rows)
                    if len(ranked) >= k or count >= min(512, len(self.model.items)):
                        break
                    count = min(count * 2, 512, len(self.model.items))
            if len(ranked) < k:
                # Scores below every retrieved result make this a fill policy,
                # not an unlabelled mixing of popularity counts with dot scores.
                floor = min(candidates.values(), default=0.0) - 1.0
                for position, item in enumerate(self.model.popular):
                    candidates.setdefault(item, floor - position / max(1, len(self.model.items)))
                ranked = rank(candidates, seen, snapshot, self.model.item_rows)
                if source == 'cpp_retrieval':
                    source = 'cpp_retrieval_with_popularity_fill'
            response = dict(schema_version=1, request_id=request_id, user_id=user,
                            model_version=self.model.version, model_sha256=self.model.fingerprint,
                            policy_version='eligible-save-rate-v1', feature_generation=snapshot['generation'],
                            candidate_source=source, degraded=source == 'popularity_upstream_unavailable',
                            last_feedback_commit_ms=snapshot['last_committed_ms'],
                            items=[dict(item_id=item, score=score) for item, score in ranked[:k]],
                            exhausted=len(ranked) < k)
            try:
                return self.store.publish(response, k)
            except Changed:
                continue
        raise Changed('feedback changed during all three retrieval attempts; retry with the same request ID')


class Server(ThreadingHTTPServer):
    daemon_threads = False
    request_queue_size = 16

    def __init__(self, address, pilot):
        if address[0] != '127.0.0.1':
            raise ValueError('pilot is loopback only; not a public HTTP server')
        self.pilot = pilot
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(2)
        return sock, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # Do not put user identities or event payloads in access logs.

    def reply(self, status, value):
        data = json.dumps(value, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def allowed(self):
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        return self.headers.get('Host') in hosts and not self.headers.get('Origin')

    def do_GET(self):
        if not self.allowed():
            return self.reply(403, {'error': 'local non-browser client required'})
        if self.path != '/healthz':
            return self.reply(404, {'error': 'unknown endpoint'})
        try:
            generation = self.server.pilot.store.snapshot('')['generation']
            self.reply(200, dict(status='ok', feature_generation=generation,
                                 scope='local policy/database; not upstream readiness'))
        except sqlite3.Error:
            self.reply(503, {'error': 'feature database unavailable'})

    def do_POST(self):
        try:
            if not self.allowed():
                return self.reply(403, {'error': 'local non-browser client required'})
            if self.path not in ('/v1/recommend', '/v1/events'):
                return self.reply(404, {'error': 'unknown endpoint'})
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError('exactly one Content-Length and no transfer encoding required')
            length = int(lengths[0])
            if not 0 < length <= 8192 or self.headers.get('Content-Type') != 'application/json':
                raise ValueError('JSON content type and body size 1..8192 required')
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('incomplete request body')
            payload = json.loads(body)
            if self.path == '/v1/recommend':
                result = self.server.pilot.recommend(payload)
            else:
                result = self.server.pilot.store.apply(payload)
            self.reply(200, result)
        except Conflict as exc:
            self.reply(409, {'error': str(exc)})
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            self.reply(400, {'error': str(exc)})
        except (Changed, sqlite3.Error):
            self.reply(503, {'error': 'state busy or unavailable; retry the same ID'})
        except (TimeoutError, OSError):
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--port', type=int, default=9410)
    parser.add_argument('--upstream-port', type=int, default=9400)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535 or args.port == args.upstream_port:
        parser.error('distinct valid ports required')
    args.db.parent.mkdir(parents=True, exist_ok=True)
    model = Model(args.manifest)
    pilot = Pilot(model, Store(args.db, model.fingerprint), Upstream(args.upstream_port, model.items))
    with Server(('127.0.0.1', args.port), pilot) as server:
        print(json.dumps(dict(ready=True, port=args.port, model_sha256=model.fingerprint)), flush=True)
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
