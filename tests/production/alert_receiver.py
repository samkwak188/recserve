"""Test-only webhook sink: records attempts and rejects the first HTTPS outage notification."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

attempts = []
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != '/attempts':
            self.send_error(404)
            return
        with lock:
            body = json.dumps(attempts).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/alerts':
            self.send_error(404)
            return
        size = int(self.headers.get('Content-Length', 0))
        if not 0 < size <= 65536:
            self.send_error(413)
            return
        data = json.loads(self.rfile.read(size))
        with lock:
            readiness_failure = any(a['labels']['alertname'] == 'RecServeHTTPSUnavailable'
                                    and a['status'] == 'firing' for a in data['alerts'])
            accepted = not readiness_failure or any(not item['accepted'] for item in attempts)
            attempts.append(dict(received_ms=time.time_ns() // 1000000,
                accepted=accepted, status=data['status'],
                alerts=[dict(status=a['status'], labels=a['labels'],
                    fingerprint=a['fingerprint'], starts_at=a['startsAt']) for a in data['alerts']]))
        self.send_response(200 if accepted else 503)
        self.send_header('Content-Length', '0')
        self.end_headers()


ThreadingHTTPServer(('0.0.0.0', 8088), Handler).serve_forever()
