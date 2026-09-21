"""Bounded RSV1 client. Do not interpret a partial or malformed reply as success."""
import math
import socket
import struct
import time


class Upstream:
    def __init__(self, port, item_ids, timeout=0.25):
        self.port, self.item_ids, self.timeout = port, item_ids, timeout

    def retrieve(self, user_row, count):
        deadline = time.monotonic() + self.timeout
        with socket.create_connection(('127.0.0.1', self.port), timeout=self.timeout) as sock:
            def exact(n):
                output = bytearray()
                while len(output) < n:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError('upstream deadline')
                    sock.settimeout(left)
                    part = sock.recv(n-len(output))
                    if not part:
                        raise OSError('upstream disconnected')
                    output.extend(part)
                return output
            sock.sendall(struct.pack('<IIQIIII', 0x52535631, 24, 1, user_row, count, count,
                                     max(1, int((deadline-time.monotonic())*1e6))))
            magic, length = struct.unpack('<II', exact(8))
            if magic != 0x52535631 or not 16 <= length <= 4112:
                raise OSError('invalid upstream framing')
            data = exact(length)
            rid, status, size = struct.unpack_from('<QII', data)
            if rid != 1 or status != 0 or size > count or length != 16 + size*8:
                raise OSError('upstream error or invalid result')
            output = {}
            for i in range(size):
                item, score = struct.unpack_from('<If', data, 16 + 8*i)
                if item >= len(self.item_ids) or not math.isfinite(score) or self.item_ids[item] in output:
                    raise OSError('invalid upstream item or score')
                output[self.item_ids[item]] = score
            return output
