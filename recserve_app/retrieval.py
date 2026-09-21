import math
import secrets
import socket
import struct
import time

MAGIC = 0x52535632


class Retrieval:
    def __init__(self, host, port, digest, dim):
        self.host, self.port, self.digest, self.dim = host, port, digest, dim
        # Resolve during paired-service startup, never on a latency-bounded request.
        # Recreating retrieval with a different address requires restarting its API pair.
        self.addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)

    def connect(self, deadline):
        error = OSError('No retrieval address')
        for family, kind, protocol, _, address in self.addresses:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Retrieval deadline expired')
            connection = socket.socket(family, kind, protocol)
            connection.settimeout(remaining)
            try:
                connection.connect(address)
                return connection
            except OSError as exc:
                connection.close()
                error = exc
        raise error

    def query(self, vector, count, deadline=None):
        deadline = deadline or time.monotonic() + .2
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Retrieval deadline expired')
        rid = secrets.randbits(64)
        with self.connect(deadline) as connection:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Retrieval deadline expired')
            payload = struct.pack('<QIII32s', rid, min(1000000, max(1, int(remaining * 1e6))),
                                  self.dim, count, bytes.fromhex(self.digest)) + struct.pack('<' + 'f' * self.dim, *vector)
            connection.settimeout(remaining)
            connection.sendall(struct.pack('<II', MAGIC, len(payload)) + payload)

            def receive(n):
                result = bytearray()
                while len(result) < n:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Retrieval deadline expired')
                    connection.settimeout(remaining)
                    part = connection.recv(n - len(result))
                    if not part:
                        raise EOFError('Retrieval connection closed')
                    result.extend(part)
                return result

            magic, size = struct.unpack('<II', receive(8))
            if magic != MAGIC or not 48 <= size <= 48 + 512 * 8 or (size - 48) % 8:
                raise ValueError('Invalid response frame')
            data = receive(size)
            actual_id, status, n, actual_model = struct.unpack_from('<QII32s', data)
            if actual_id != rid or status != 0 or actual_model.hex() != self.digest or n > count or size != 48 + 8 * n:
                raise ValueError('Retrieval rejected or mismatched request')
            items = [struct.unpack_from('<If', data, 48 + 8 * i) for i in range(n)]
            if len({item for item, _ in items}) != n or any(not math.isfinite(score) for _, score in items):
                raise ValueError('Invalid retrieval result')
            return items

    def ready(self):
        self.query([0.] * self.dim, 1)
