import time
import pytest
from recserve_app.retrieval import Retrieval


def test_requests_do_not_resolve_dns(live_model, monkeypatch):
    model, upstream = live_model
    client = Retrieval(upstream.host, upstream.port, model.digest, model.dimension)
    def forbidden(*args, **kwargs):
        raise AssertionError('DNS must not run in a recommendation request')
    monkeypatch.setattr('socket.getaddrinfo', forbidden)
    assert client.query([0.] * model.dimension, 1)
    with pytest.raises(TimeoutError):
        client.query([0.] * model.dimension, 1, deadline=time.monotonic() - 1)
