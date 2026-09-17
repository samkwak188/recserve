# Cost and capacity board

- Host: Windows-11-10.0.26200-SP0 / ARM64 / 12 logical CPUs
- Measured: 2026-09-17T19:14:24Z
- Protocol: in-process `recommend_sync`, 3 trials, 64-request warmup, catalog = synthetic unit-normalized embeddings
- This is a single-host benchmark, not production serving.

## Quality gate (data/quality_fixture.csv, k=5)

- float32: recall=0.2000, NDCG=0.1000, retrieve-recall=1.0000, users=5
- int8: recall=0.2000, NDCG=0.1000, retrieve-recall=1.0000

## Engine throughput (16384 items, dim=64, retrieve_k=64, k=10, n=600)

| mode | p99 mean us | p99 CV | QPS | RSS MiB |
|---|---|---|---|---|
| baseline | 96.36 | 0.246 | 17819 | 18.1 |
| arena | 103.73 | 0.266 | 16972 | 18.2 |
| soa | 103.10 | 0.143 | 15454 | 18.2 |
| simd | 107.46 | 0.169 | 14878 | 18.2 |
| int8 | 179.56 | 0.139 | 7875 | 19.3 |
| pin | 83.39 | 0.094 | 16035 | 18.2 |

## Same protocol, 4096-item catalog

| mode | p99 mean us | p99 CV | QPS | RSS MiB |
|---|---|---|---|---|
| baseline | 62.69 | 0.024 | 26893 | 8.9 |
| arena | 51.69 | 0.207 | 27732 | 8.9 |
| soa | 72.80 | 0.350 | 24655 | 8.9 |
| simd | 44.68 | 0.085 | 26576 | 8.9 |
| int8 | 77.03 | 0.047 | 15012 | 9.2 |
| pin | 42.00 | 0.000 | 26595 | 8.9 |

## Predeclared SLO (from baseline p99, not reverse-picked)

- Baseline p99 mean = 96.36 us on the 16384-item catalog.
- SLO = 1.10x baseline = **105.99 us p99**. SIMD mode p99 mean = 107.46 us (FAIL).

## Nearline / soak

- `events=5000 freshness_p50_ms=2550.5 freshness_p99_ms=5000.01 replica_lag_p50_ms=15 replica_lag_p99_ms=15 fault_delay_us=20000`
- `requests=103927 start_rss=7192576 max_rss=7409664 delta=217088`

Do not rewrite these as production QPS, multi-region serving, or CHTC results.
