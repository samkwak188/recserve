#!/usr/bin/env python3
"""Interleave base and candidate binaries over byte-identical fixtures."""
import argparse
import json
import math
import pathlib
import statistics
import subprocess


def verdict(base, candidate, max_ratio=1.25, max_recall_drop=.01):
    fields = ('p99_mean_us', 'response_recall', 'qps_mean')
    for rows in (base, candidate):
        if not rows or any(not all(f in row and math.isfinite(row[f]) for f in fields) for row in rows):
            return {'passed': False, 'reasons': ['missing or nonfinite measurement']}
        if any(row['p99_mean_us'] <= 0 or row['qps_mean'] <= 0 or not 0 <= row['response_recall'] <= 1 for row in rows):
            return {'passed': False, 'reasons': ['invalid latency, throughput or quality']}
    ratio = statistics.median(x['p99_mean_us'] for x in candidate) / statistics.median(x['p99_mean_us'] for x in base)
    drop = statistics.mean(x['response_recall'] for x in base) - statistics.mean(x['response_recall'] for x in candidate)
    reasons = []
    if ratio > max_ratio:
        reasons.append('p99 regression')
    if drop > max_recall_drop:
        reasons.append('recall regression')
    return {'passed': not reasons, 'reasons': reasons, 'p99_ratio': ratio, 'recall_drop': drop}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-bench', required=True)
    p.add_argument('--candidate-bench', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--out', default='.cache/logs/regression.json')
    args = p.parse_args()
    if args.repeats < 3:
        p.error('at least three independent paired runs are required')
    root = pathlib.Path(__file__).resolve().parents[1]
    scratch = root / '.cache' / 'gate'
    scratch.mkdir(parents=True, exist_ok=True)
    catalog, index = scratch / 'catalog.bin', scratch / 'index.bin'
    subprocess.run([args.fixture, '--items', '16384', '--dim', '64', '--clusters', '256',
                    '--build-threads', '1', '--out-catalog', str(catalog), '--out-index', str(index)], check=True)
    rows = {'base': [], 'candidate': []}
    binaries = dict(base=args.base_bench, candidate=args.candidate_bench)
    for repeat in range(args.repeats):
        for side in (('base', 'candidate') if repeat % 2 == 0 else ('candidate', 'base')):
            command = [binaries[side], '--load-catalog', str(catalog), '--load-index', str(index),
                       '--kernel', 'simd', '--n', '3000', '--trials', '3', '--recall-probe', '128', '--json']
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            rows[side].append(json.loads(result.stdout.strip().splitlines()[-1]))
    result = verdict(rows['base'], rows['candidate'])
    result['measurements'] = rows
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'measurements'}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
