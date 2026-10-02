"""A cooperative asyncio timer must not turn a late response into success."""
import argparse
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import open_loop


class Reader:
    def __init__(self, delay, status, clock):
        self.delay, self.status, self.clock = delay, status, clock
        self.reads = 0

    async def readexactly(self, size):
        self.reads += 1
        if self.reads == 1:
            return struct.pack('<II', 0x52535631, 16)
        # Advance the observed completion clock without yielding to the timer.
        # Keep the event loop's own clock real so scheduling speed cannot
        # determine whether this regression reaches the response body.
        self.clock[0] += self.delay
        return struct.pack('<QII', 0, self.status, 0)


class Writer:
    def write(self, data):
        pass

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def measure(self, delay, status, deadline):
        clock = [0.0]
        reader = Reader(delay, status, clock)
        async def connect(*args):
            return reader, Writer()
        args = argparse.Namespace(host='test-only', port=1, rate=1, seconds=1,
            concurrency=1, queue=1, deadline_ms=deadline, users=1, k=10, retrieve_k=64)
        with patch.object(open_loop.asyncio, 'open_connection', connect), \
             patch.object(open_loop, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
            result = await open_loop.campaign(args)
        self.assertEqual(reader.reads, 2)
        return result

    async def test_overdue_timer_cannot_accept_late_success(self):
        result = await self.measure(.03, 0, 5)
        self.assertEqual(result['counts'], {'client_deadline': 1})
        self.assertEqual(result['failure_rate'], 1)
        self.assertEqual(result['latency_samples'], 0)
        self.assertIsNone(result['success_p99_us'])

    async def test_late_error_is_also_a_client_deadline(self):
        result = await self.measure(.03, 1, 5)
        self.assertEqual(result['counts'], {'client_deadline': 1})

    async def test_on_time_success_and_server_failure_remain_distinct(self):
        for status, expected in ((0, 'ok'), (1, 'server_timeout')):
            with self.subTest(status=status):
                result = await self.measure(.001, status, 1000)
                self.assertEqual(result['counts'], {expected: 1})
                self.assertEqual(result['latency_samples'], int(status == 0))


if __name__ == '__main__':
    unittest.main()
