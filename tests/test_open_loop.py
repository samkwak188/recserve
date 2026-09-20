import argparse
import asyncio
import importlib.util
import pathlib
import socket
import unittest

spec = importlib.util.spec_from_file_location('open_loop', pathlib.Path(__file__).resolve().parents[1]/'scripts/open_loop.py')
load = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load)


class LoadTests(unittest.TestCase):
    def test_failure_accounting(self):
        # Hold a bound but non-listening port: every connection must fail.
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            args = argparse.Namespace(host='127.0.0.1', port=s.getsockname()[1], rate=1000,
                                      seconds=.02, queue=1, concurrency=1, deadline_ms=100,
                                      users=1, k=1, retrieve_k=1)
            result = asyncio.run(load.campaign(args))
        self.assertEqual(sum(result['counts'].values()), result['attempted'])
        self.assertEqual(result['failure_rate'], 1)
        self.assertIsNone(result['success_p99_us'])


if __name__ == '__main__':
    unittest.main()
