import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('gate', pathlib.Path(__file__).resolve().parents[1] / 'scripts/regression_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTests(unittest.TestCase):
    def test_negative_gates(self):
        base = [dict(p99_mean_us=100, qps_mean=20000, response_recall=.95)] * 3
        self.assertTrue(gate.verdict(base, base)['passed'])
        self.assertFalse(gate.verdict([], base)['passed'])
        for changes in (dict(p99_mean_us=200), dict(response_recall=.8), dict(qps_mean=0), dict(p99_mean_us=float('nan'))):
            candidate = [dict(base[0], **changes)] * 3
            self.assertFalse(gate.verdict(base, candidate)['passed'])


if __name__ == '__main__':
    unittest.main()
