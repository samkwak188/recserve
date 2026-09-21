import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('production_runner', Path(__file__).resolve().parents[1] / 'scripts/production_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExecutionTests(unittest.TestCase):
    def test_log_and_negative_cache_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / 'step.log'
            self.assertEqual(runner.execute([sys.executable, '-c', 'print("evidence")'], log, 5), 0)
            record = dict(fingerprint='a', status='passed', steps=[dict(exit_code=0, log=log.name, log_sha256=runner.digest(log))])
            self.assertTrue(runner.reusable(record, 'a', root))
            self.assertFalse(runner.reusable(record, 'b', root))
            log.write_text('tampered')
            self.assertFalse(runner.reusable(record, 'a', root))
            self.assertFalse(runner.reusable(dict(fingerprint='a', status='passed'), 'a', root))

    def test_timeout_and_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / 'step.log'
            self.assertEqual(runner.execute([sys.executable, '-c', 'raise SystemExit(7)'], log, 5), 7)
            self.assertEqual(runner.execute([sys.executable, '-c', 'import time; time.sleep(30)'], log, 1), 124)

    def test_fingerprint_tracks_untracked_changed_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            p = root / 'source'
            p.write_text('one')
            subprocess.run(['git', 'add', 'source'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], cwd=root, check=True)
            first = runner.source_identity(root)
            p.write_text('two')
            self.assertNotEqual(first, runner.source_identity(root))
            p.unlink()
            deleted = runner.source_identity(root)
            (root / 'new').write_text('three')
            self.assertNotEqual(deleted, runner.source_identity(root))
