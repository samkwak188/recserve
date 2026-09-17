# RecServe

**A C++20 two-stage recommendation serving engine you can run, measure, and break.**

RecServe answers one request: given a user, retrieve a few hundred candidate items from an embedding index, score them with a linear ranker plus nearline features, and return the top-K. It is infrastructure for recommendation **serving**, not a trained TikTok model.

```
user request
    │
    ├─ feature store   (recent items, item CTR, optional injected delay)
    ├─ retrieve        (NSW / sampled graph, brute-force oracle for recall)
    └─ rank            (dot product + feature weights, SoA / NEON / int8 knobs)
         │
         └─ top-K + latency histograms + quality gate
```

It is a **single-host benchmarked prototype**. It is not production, not multi-region, and not ByteDance-scale.

## What it is for

| You want | RecServe does |
|---|---|
| C++ in the body of a rec-infra resume | A real retrieve→rank service, load generator, and tests |
| Performance-efficiency evidence | p50/p95/p99, QPS, RSS, 3-trial CV, a predeclared SLO |
| Quality under optimization | recall/NDCG gate plus retrieve-recall vs brute force |
| Streaming features | Kafka-shaped event records, freshness, fault delay |
| Reliability | timeouts, load-shed, soak RSS, bottleneck classifier |

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure   # Linux
build\Release\recserve_tests.exe             # Windows
python scripts/measure.py                    # writes results/measured.json and COST.md
```

Serve and hit it:

```bash
build/Release/recserve_server --port 9400 --items 4096 --dim 64
build/Release/recserve_loadgen --port 9400 --qps 200 --n 500
```

## Measured results (this machine)

Host: **Windows 11 ARM64, 12 logical CPUs**, 2026-09-17. Protocol: in-process `recommend_sync`, 3 trials, 64 warmup requests, synthetic unit-normalized embeddings. Raw dump: [`results/measured.json`](results/measured.json). Board: [`COST.md`](COST.md).

**16,384 items × dim 64, retrieve 64, return 10, 600 requests**

| mode | p99 mean | QPS | RSS |
|---|---|---|---|
| baseline (AoS, heap) | 96.4 µs | 17,819 | 18.1 MiB |
| worker pinning | **83.4 µs** | 16,035 | 18.2 MiB |
| int8 embeddings | 179.6 µs | 7,875 | 19.3 MiB |

Predeclared SLO = 1.10 × baseline p99 = **106.0 µs**. Pinning passes. The ARM64 NEON scorer did not beat baseline on this catalog (107.5 µs) — that result is left in the table on purpose.

**Quality gate** on `data/quality_fixture.csv` (k=5): retrieve-recall vs brute force = **1.0**. Interaction recall@5 = 0.20 because item embeddings are random, not trained on the log. Int8 did not change that recall on this fixture.

**Soak:** 103,927 requests in 3 s, RSS +217 KiB.

**Nearline:** 5,000 events, freshness p50 = 2.55 s / p99 = 5.00 s against synthetic timestamps; replica lag knob = 15 ms; feature-store delay injection = 20 ms.

## Binaries

| binary | role |
|---|---|
| `recserve_server` | TCP retrieve-rank service |
| `recserve_loadgen` | open-loop client at a fixed arrival rate |
| `recserve_bench` | in-process p99/QPS/RSS (`--json --mode baseline\|arena\|soa\|simd\|int8\|pin`) |
| `recserve_quality` | recall / NDCG / retrieve-recall (`--json --int8`) |
| `recserve_nearline` | event log → CTR / recency / freshness |
| `recserve_rankd` | ranker process with a bounded queue and load-shed |
| `recserve_soak` | leak/RSS soak (`--seconds 86400` for 24 h) |
| `recserve_diagnose` | observe → classify bottleneck → print the knob to retune |
| `recserve_tests` | scorer, protocol, top-K, timeout, load-shed, nearline |

## Workloads

- **Quality track:** temporal split on an interaction CSV. Official [KuaiRec](https://kuairec.com/) (CC BY-SA 4.0) is 1,411 users / 3,327 items (small) and 7,176 / 10,728 (big). Convert with `python scripts/prep_kuairec.py`. Do not use KuaiRec as the QPS catalog — it is too small for ANN to matter.
- **Scale track:** synthetic embeddings. 16,384×64 is the published board. `python scripts/gen_scale.py --n 100000` builds a larger fixture.

## Kafka

Events are 17 bytes: `event_time_ms, user_id, item_id, type`. Tests and `recserve_nearline` use a file. `docker compose up` runs Redpanda; `python scripts/kafka_produce.py` writes the same records. C++ librdkafka is optional (`-DRECSERVE_WITH_RDKAFKA=ON`). There is no Flink job.

## License

MIT. See [LICENSE](LICENSE).
