import json
import time
import uuid
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
from sqlalchemy.exc import TimeoutError as PoolTimeout, DBAPIError


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter('recserve_http_requests_total', 'Completed HTTP requests', ['route', 'method', 'status'], registry=self.registry)
        self.latency = Histogram('recserve_http_duration_seconds', 'Server HTTP latency', ['route'],
            buckets=(.001, .005, .01, .025, .05, .075, .1, .2, .5, 1, 2, 5), registry=self.registry)
        self.active = Gauge('recserve_http_active', 'In-flight HTTP requests', registry=self.registry)
        self.recommendations = Counter('recserve_recommendations_total', 'Recommendation policy outcomes', ['source', 'degraded'], registry=self.registry)

        self.database_errors = Counter('recserve_database_errors_total',
            'HTTP failures caused by the database', ['reason'], registry=self.registry)
        self.pool_used = Gauge('recserve_database_pool_checked_out',
            'Connections currently checked out by this API process', registry=self.registry)
        self.pool_limit = Gauge('recserve_database_pool_capacity',
            'Maximum connections allowed by this API process', registry=self.registry)
        self.privacy_ready = Gauge('recserve_privacy_ready',
            'Last completed privacy reconciliation succeeded (not a live ledger probe)', registry=self.registry)
        self.retrieval = Histogram('recserve_retrieval_duration_seconds',
            'Retrieval wall time including connect, send and receive; includes readiness probes',
            ['outcome'], buckets=(.001, .005, .01, .025, .05, .1, .2, .3, 1), registry=self.registry)


def database_failure_reason(exc):
    # Bounded labels only: never exception messages, SQL, URLs or account data.
    if isinstance(exc, PoolTimeout):
        return 'pool_timeout'
    if isinstance(exc, DBAPIError):
        state = getattr(exc.orig, 'sqlstate', None)
        if state in ('55P03', '57014', '40001', '40P01'):
            return {'55P03': 'lock_timeout', '57014': 'query_cancelled',
                    '40001': 'serialization', '40P01': 'deadlock'}[state]
        if exc.connection_invalidated or (state and state.startswith('08')):
            return 'connection'
    return 'other'


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
