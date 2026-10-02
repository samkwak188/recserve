"""Trace-integrity gates run in hosted CPU CI; no GPU availability is implied."""
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.profile_gpu import timeline


class TraceTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory) / 'trace.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript("""
CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,demangledName INTEGER);
CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
INSERT INTO StringIds VALUES(1,'cudaMemcpyAsync_v3020'),(2,'cudaLaunchKernel_v7000'),(3,'cudaStreamSynchronize_v3020'),(4,'kernel');
""")
            for i in range(3):
                start, correlation = i * 10000, i + 10
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)',
                    [(start, start+10, 7, i, 1), (start+100, start+500, 7, correlation, 2),
                     (start+800, start+900, 7, i+20, 3)])
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,4)',
                    (start+600, start+700, correlation))
                db.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,7,?,NULL)', (start+20, start+600, 'heuristic'))
            # Unrelated activity within a request must not inflate its device evidence.
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(650,660,999,4)')
        return path

    def test_correlated_kernels_and_nested_host_ranges_stay_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = timeline(self.fixture(directory))
            self.assertEqual(result['kernel_events'], 4)
            for episode in result['episodes']:
                self.assertEqual(len(episode['kernels']), 1)
                self.assertEqual(episode['kernel_duration_sum_ms'], .0001)
                self.assertEqual(episode['library_ranges'][0]['duration_ms'], .00058)
                self.assertEqual(episode['first_copy_to_sync_ms'], .0009)

    def test_empty_capture_and_missing_episode_fail(self):
        for statement in ('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL',
                          'DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= 20000'):
            with self.subTest(statement=statement), tempfile.TemporaryDirectory() as directory:
                path = self.fixture(directory)
                with sqlite3.connect(path) as db:
                    db.execute(statement)
                with self.assertRaises(RuntimeError):
                    timeline(path)

    def test_inconsistent_clock_interval_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory)
            with sqlite3.connect(path) as db:
                db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=9999 WHERE correlationId=10')
            with self.assertRaises(RuntimeError):
                timeline(path)

    def test_missing_file_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.sqlite'
            with self.assertRaises(sqlite3.OperationalError):
                timeline(path)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
