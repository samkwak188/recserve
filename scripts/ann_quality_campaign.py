#!/usr/bin/env python3
"""Characterize a high-recall ANN operating point on the same frozen GPU fixture."""
import json
import pathlib
import statistics
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
rows = []
for ef in (384, 1024, 4096, 16384):
    group = []
    for repeat in range(3):
        command = [str(root/'build-gpu/recserve_gpu_bench'), '--catalog', str(root/'data/gpu_catalog_1000000.bin'),
                   '--index', str(root/'data/gpu_index_1000000.bin'), '--backend', 'hnsw', '--batch', '1',
                   '--iterations', '100', '--ef', str(ef)]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        row = json.loads(result.stdout.strip().splitlines()[-1]); row['repeat'] = repeat
        rows.append(row); group.append(row)
    print(json.dumps(dict(ef=ef, recall_min=min(r['recall'] for r in group),
                         qps_median=statistics.median(r['qps'] for r in group))), flush=True)
    if min(r['recall'] for r in group) >= .99:
        break
source = json.loads((root/'results/gpu-crossover.json').read_text())
payload = dict(protocol='same frozen 1M fixture and independent 128-query quality probes as gpu-crossover; target recall>=.99',
               provenance=source['provenance'], fixture=source['fixtures']['1000000'], measurements=rows)
(root/'results/ann-high-recall.json').write_text(json.dumps(payload, indent=2)+'\n')
