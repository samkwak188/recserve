import json
import time
import uuid
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter('recserve_http_requests_total', 'Completed HTTP requests', ['route', 'method', 'status'], registry=self.registry)
        self.latency = Histogram('recserve_http_duration_seconds', 'Server HTTP latency', ['route'],
            buckets=(.001, .005, .01, .025, .05, .075, .1, .2, .5, 1, 2, 5), registry=self.registry)
        self.active = Gauge('recserve_http_active', 'In-flight HTTP requests', registry=self.registry)
        self.recommendations = Counter('recserve_recommendations_total', 'Recommendation policy outcomes', ['source', 'degraded'], registry=self.registry)


class Telemetry:
    def __init__(self, app, metrics):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        started, status = time.monotonic(), 500
        trace = uuid.uuid4().hex
        self.metrics.active.inc()

        async def observe(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                message.setdefault('headers', []).append((b'x-request-id', trace.encode()))
            await send(message)
        try:
            await self.app(scope, receive, observe)
        finally:
            elapsed = time.monotonic() - started
            route = getattr(scope.get('route'), 'path', 'unmatched')
            method = scope['method'] if scope['method'] in ('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS') else 'OTHER'
            self.metrics.active.dec()
            self.metrics.latency.labels(route).observe(elapsed)
            self.metrics.requests.labels(route, method, str(status)).inc()
            if route not in ('/healthz', '/readyz', '/internal/metrics'):
                print(json.dumps(dict(timestamp_ms=time.time_ns() // 1000000, route=route, method=method,
                    status=status, duration_ms=round(elapsed * 1000, 3), request_id=trace)), flush=True)
