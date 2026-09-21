# Historical engine benchmarks and experiments

This preserves the earlier engine overview and measurements. Some architecture
descriptions below concern separate experiments, not a connected deployment.
Start with [README.md](README.md) for the supported local workflow.

**A C++20 recommendation serving engine you can run, measure, and break.**

## Current checkpoint: 2026-09-20

CUDA retrieval now runs on an RTX 3060: persistent catalog/workspace, tiled FP32
cuBLAS scoring, device-side top-K, deadline-aware microbatching and exact CPU
fallback. The internal TCP service now has bounded admission, I/O deadlines,
health/metrics, graceful shutdown and verified real-data bundles. A non-root,
read-only CPU container has served the trained model successfully.

All **16 expanded hosted CI jobs** passed for runtime checkpoint `7e0eb63`,
including Windows/Linux, sanitizers, CUDA compilation, Kafka and Flink:
[open the run](https://github.com/samkwak188/recserve/actions/runs/35541070732).
Actual CUDA tests ran locally on the RTX; hosted CUDA CI is compile-only.
Compute Sanitizer remains blocked by the Windows GPU debugger setting and is
not claimed as passing. No persistent self-hosted GPU runner was registered.

Start with [the execution plan and remaining release gates](EXECUTION_PLAN.md),
[local run/rollback instructions](OPERATIONS.md),
[the new GPU/service measurements](results/EXECUTION_REPORT.md), and
[the separate x86 CPU board](results/WSL_CPU.md).

This is a hardened **local prototype**, not yet production: durable feature
recovery and the event-to-server bridge, serving-time seen/availability filters,
a learned ranker, authentication, long-duration soak, and an opt-in real-user
pilot remain open. The original ARM results below are historical, not RTX/WSL
measurements. Never mix the two hosts in a performance claim.

RecServe answers one request: given a user, retrieve a few hundred candidates
from an HNSW index, rank them with a linear scorer over nearline features, and
return the top-K. Around that sit the parts that make a serving system a system
rather than a function — a streaming feature pipeline, a lock-free snapshot, a
regression gate, a capacity model, and a closed-loop tuner that is not allowed
to claim a win it cannot measure.

```
events (Kafka / file)                     query
        │                                   │
        ▼                                   ▼
  nearline ingest                    ┌─ feature read ── RCU snapshot, no lock
  windowed CTR                       │
        │ publish on an interval     ├─ retrieve ───── HNSW, ef-tunable
        ▼                            │
  feature snapshot  ─────────────────┤
        │                            └─ rank ───────── dot + features
        │                                   │            scalar/SIMD/int8/AoSoA
        ▼                                   ▼
  Flink batch aggregate            top-K + p50/p95/p99 + recall
        │                                   │
        └──────── skew measured ────────────┘
                                            │
                              closed-loop agent: observe, diagnose,
                              plan, execute, verify, roll back
```

It is a **single-host benchmarked prototype**. It is not production, not
multi-region, and not ByteDance scale.

## What is actually measured

Full board: [`COST.md`](COST.md). Raw: [`results/`](results/). Regenerate with
`python scripts/measure.py`. Host below: Windows 11 ARM64 (Snapdragon, 12
logical CPUs), ISA `neon+sdot`, 2026-09-17.

### The index is correct, and that is checked against the reference

RecServe implements HNSW from [Malkov & Yashunin,
arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — Algorithms 1, 2, 4 and 5,
level assignment with mL = 1/ln(M), M_max0 = 2M, and the neighbour-selection
heuristic on forward *and* reverse links.
[`scripts/validate_hnsw.py`](scripts/validate_hnsw.py) runs it against
[hnswlib](https://github.com/nmslib/hnswlib) over the same catalog, the same
queries and matched (M, efConstruction, ef), both scored against exact numpy
top-k:

| ef | hnswlib recall@10 | RecServe recall@10 | delta |
|---|---|---|---|
| 32 | 0.3684 | 0.3742 | +0.0059 |
| 64 | 0.5445 | 0.5426 | −0.0020 |
| 128 | 0.7285 | 0.7273 | −0.0012 |
| 256 | 0.8902 | 0.8820 | −0.0082 |

Max delta **0.0082**. hnswlib reports the same recall on this data, so the
absolute values are a property of 64-dim near-uniform vectors, not of the
implementation.

### Kernels: where layout and dtype actually bind

Exact scan, 1,000,000 items × dim 64 (244 MiB, past every cache):

| kernel | p99 | QPS | catalog |
|---|---|---|---|
| scalar | 27.66 ms | 37 | 244 MiB |
| **int8** | **5.26 ms** | **197** | 309 MiB |
| simd | 5.96 ms | 175 | 244 MiB |
| blocked (AoSoA) | 7.41 ms | 141 | 488 MiB |
| soa_strided | 51.04 ms | 29 | 488 MiB |

int8 uses a real integer dot product — NEON `SDOT` (runtime-detected via
`PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE`, falling back to widening `vmull_s8`) or
AVX2 sign-extend + `PMADDWD` — with the query quantized once per request. It is
**5.3× faster than scalar and the fastest kernel**.

`soa_strided` is kept on the board on purpose: it strides one item at a time and
is **8.6× slower than SIMD**. "Use SoA" is not advice; matching the layout to
the access pattern is.

**Blocked AoSoA has a crossover.** At 200,000 items it is 0.55× (faster) at dim
8 and 1.35× (slower) at dim 128. Below the crossover the per-item horizontal
reduction dominates and blocking amortises it across 8 items; above it,
broadcasting `q[d]` once per dimension costs more than the reduction it removes.

### The parallel index build is not reproducible, and that is why fixtures exist

Two builds of the same catalog with the same parameters produce different
graphs: with several threads inserting at once, the order reverse links arrive
changes which ones the pruning heuristic keeps. Measured on 16,384 items,
recall@10 over five builds:

| build | recall spread |
|---|---|
| 12 threads | 0.0172 |
| 1 thread | 0.0000 |
| loaded snapshot | 0.0000 |

hnswlib has the same property for the same reason. It matters because that
spread sits underneath every recall comparison that rebuilds — so every
published measurement loads an index snapshot instead, and
`--build-threads 1` is available when reproducibility is worth more than build
time. A unit test pins the single-threaded case.

### Latency is bought with recall, so both are reported

ef sweep, 65,536 items — this is a curve, not a number:

| ef | p99 | QPS | recall@10 |
|---|---|---|---|
| 16 | 36.6 µs | 43,010 | 0.235 |
| 64 | 99.4 µs | 14,224 | 0.545 |
| 128 | 201.7 µs | 8,068 | 0.713 |
| 384 | 538.7 µs | 2,743 | 0.942 |

Recall is measured **end to end**, through `recommend_sync`, so shrinking
`retrieve_k` to buy latency shows up as lost accuracy instead of hiding from a
probe that queried the index directly.

### Recommendation quality on real embeddings

Synthetic vectors measure kernels fine — a dot product does not care where the
numbers came from — but they make recall@10 a statement about high-dimensional
geometry rather than about recommendation quality.
[`scripts/prep_movielens.py`](scripts/prep_movielens.py) fits implicit-feedback
ALS ([Hu, Koren & Volinsky, ICDM 2008](https://doi.org/10.1109/ICDM.2008.22)) on
MovieLens with a per-user temporal split, and
[`recserve_eval`](apps/eval.cpp) serves those factors through the real request
path, filtering items the user already saw.

**ml-25m** — 162,342 users, 40,858 items, 10.0M train / 2.4M held out, 4,096
evaluation users, recall@10 ceiling 0.652:

| method | recall@10 | ndcg@10 |
|---|---|---|
| random | 0.0003 | 0.0006 |
| most-popular | 0.0503 | 0.0630 |
| **ALS, exact scan** | **0.0720** | **0.0739** |
| ALS, served through HNSW (ef=16) | 0.0714 | 0.0736 |

ALS beats most-popular by **+43.0%** recall — the baseline that matters, because
a model that ties popularity has learned popularity. The C++ service and an
independent numpy implementation of the same metric **agree to 0.00000**.

And the headline: **on trained embeddings HNSW reaches retrieve-recall 0.9946 at
ef=16**, versus 0.55 at ef=64 on synthetic uniform vectors. End-to-end quality
goes 0.0720 → 0.0714 — 0.8% of the model's accuracy traded for **2.0× lower p99**
(405 µs → 198 µs). Real embedding tables have low intrinsic dimension; i.i.d.
Gaussian directions are the documented worst case, and the old numbers were
measuring that rather than the graph.

### Sharding costs tail latency and buys recall

Real catalogs do not fit on one machine. [`recserve_shard`](apps/shard.cpp)
splits the catalog across N shards, scatters every query, and merges the partial
top-K lists. A request finishes when its **slowest** shard replies, so
end-to-end latency is the maximum of N samples — [Dean & Barroso, "The Tail at
Scale", CACM 56(2), 2013](https://doi.org/10.1145/2408776.2408794).

1M items, k=10, ef=64, on a 12-core host:

| shards | items/shard | shard p99 | e2e p99 | amplification | recall@10 |
|---|---|---|---|---|---|
| 1 | 1,000,000 | 260 µs | 310 µs | 1.19× | 0.1601 |
| 2 | 500,000 | 269 µs | 389 µs | 1.45× | 0.2480 |
| 4 | 250,000 | 256 µs | 451 µs | 1.76× | 0.3718 |
| 8 | 125,000 | 276 µs | 621 µs | 2.25× | 0.5003 |
| 16 \* | 62,500 | 453 µs | 3,239 µs | 7.15× | 0.6484 |
| 32 \* | 31,250 | 240 µs | 2,606 µs | 10.86× | 0.7985 |

\* more shards than cores, so those rows include CPU queueing on top of the
structural effect — read them as an upper bound.

Per-shard p99 stays flat while end-to-end p99 grows: that is the tail effect,
not slower shards. Recall rises because each shard searches its own smaller
graph at the same ef and every shard contributes its own top-k, so candidates
examined scale with N. **Sharding buys accuracy and costs tail latency** — that
trade is the decision, not whether scatter-gather works.

Honest scope: shards are threads in one process. No network, no separate failure
domain, so this is a *lower bound* on what a real deployment would see.

### The feature writer no longer stalls the readers

`FeatureStore` took one `std::mutex` on every `user()` and `item()` call.
[`feature_snapshot.hpp`](include/recserve/feature_snapshot.hpp) replaces it with
read-copy-update (McKenney & Slingwine, PDCS 1998): readers take one acquire
load, the ingest thread publishes deltas on an interval, and the grace period is
explicit — readers publish the generation they are inside and the writer waits
for quiescence. 8 serving threads, median of 3:

| events/s | store | serve QPS | p50 | p99 | freshness p99 |
|---|---|---|---|---|---|
| 10,000 | mutex | 158,858 | 37 µs | 87 µs | 0 ms |
| 10,000 | **snapshot** | 169,228 | 36 µs | **55 µs** | 104 ms |
| 500,000 | mutex | 139,807 | 40 µs | 139 µs | 0 ms |
| 500,000 | **snapshot** | 155,413 | 39 µs | **51 µs** | 113 ms |

p50 barely moves and p99 improves 1.6–2.7×, which is the signature of lock
contention rather than of less work being done. The mutex's zero freshness is
real: it is what the lock buys. Sweeping the publish interval at 500k events/s
holds serving p99 flat at ~52 µs while freshness p99 goes 25 ms (10 ms interval)
to 504 ms (500 ms interval).

### Online and offline features disagree, and by how much

The online path aggregates incrementally; training reads a batch aggregate. The
gap is training/serving skew, and it is measured rather than assumed
([`scripts/skew.py`](scripts/skew.py)), over 200k Zipf-distributed events:

| publish interval | unpublished | items differing | max ΔCTR | mean ΔCTR |
|---|---|---|---|---|
| 100 ms | 99 | 74 | 1.000 | 0.000196 |
| 30 s | 19,999 | 5,065 | 1.000 | 0.019699 |

Mean skew tracks the interval linearly. The maximum does not — it pins at 1.0 at
every interval, because an item whose only interaction is still staged reads as
CTR 0 online and 1.0 offline. **The skew budget is spent almost entirely on the
cold tail**, which is where new content lives.

### Capacity and dollars

1M qps, 100M items, 3 replicas, 65% utilisation, on-demand list prices
([`cost/instances.json`](cost/instances.json), dated and sourced):

| kernel | instance | QPS/core | B/item | hosts | $/M requests |
|---|---|---|---|---|---|
| scalar | c7g.4xlarge | 7,518 | 388 | 13 | $0.0021 |
| simd | c7g.4xlarge | 14,104 | 388 | 7 | $0.0011 |
| **int8** | c7g.4xlarge | 26,008 | 200 | **4** | **$0.0006** |

int8 takes the fleet from 7 hosts to 4 — **$2,964 → $1,694/month, saving
$1,270** — at a cost of 0.0109 recall@10. The model reports which constraint
binds (compute here, not memory) because a quantization that halves memory buys
nothing on a compute-bound fleet.

### The agent, and what it refused to claim

[`scripts/agent.py`](scripts/agent.py) runs observe → diagnose → plan → execute
→ verify → improve. The planner is swappable — heuristic rules, an LLM, random,
or grid — and `--compare` runs them from the same start on the same budget,
because a tuner that cannot beat random search has not earned its inference
cost.

From a misconfigured start (ef=256, retrieve_k=128, scalar), 12 iterations:

| planner | start p99 | best p99 | improvement | confirmed |
|---|---|---|---|---|
| **heuristic** | 656.5 µs | **230.4 µs** | **+64.9%** | yes |
| random | 841.2 µs | 351.4 µs | +58.2% | yes |
| grid | 629.9 µs | 637.6 µs | −1.2% | no |

Grid fails at equal budget because fixed-order enumeration spends all 12
iterations on `ef` and never reaches the kernel knob — which is where the 2.8×
was. That is the argument for a diagnosis-driven planner over a sweep.

**On an already-tuned config, all planners accept nothing.** The best remaining
change is `retrieve_k` 64 → 16; a separate 40-run interleaved A/B measures it at
+2.9% with t = 2.10, below this host's detection floor. Declining to claim it is
the correct outcome — and the first two stages of the gate had accepted it.

Getting the verifier right took four corrections, each forced by a measurement:

1. Rebuilding the index per observation made baseline p99 swing 128–172 µs.
   Observations now load a prebuilt snapshot.
2. The gate used the within-process trial SD as the uncertainty of a *mean* over
   trials. That is the spread of the samples, not the error of their average.
3. Within-process spread says nothing about comparing two separate processes.
   Measured on one fixed config, between-run CV is 5.7% at n=600 and 2.6% at
   n=5000 — a smallest-detectable-change of 11.4% and 5.2%. The gate now
   **calibrates σ by running the same config five times before it will accept
   anything**, and re-measures the incumbent beside every candidate.
4. Even then a no-op slipped through, because `ef` below `retrieve_k` is a no-op
   (the engine searches with `max(ef, retrieve_k)`) and 12 comparisons at 2σ
   produce false positives by construction. Wins must now **replicate**, and a
   final independent A/B confirms the run.

**The LLM never writes code, runs a command, or touches the serving path.** It
returns one `(knob, value)` pair, validated against an enumerated action space;
anything unparseable or out-of-space is discarded and the heuristic planner
takes the turn. The backend sits behind one interface, so `--provider claude` or
`gemini` changes who proposes and nothing about what gets accepted.

## Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --output-on-failure    # Linux
build\Release\recserve_tests.exe              # Windows

# build a catalog + index once, then measure against the snapshot
build/Release/recserve_fixture --items 1000000 --dim 64 --clusters 4096 \
    --out-catalog data/catalog_1000k.bin --out-index data/index_1000k.bin

python scripts/measure.py          # full campaign -> results/ + COST.md
python scripts/measure.py --quick  # the cheap stages only
```

Serve and hit it:

```bash
build/Release/recserve_server --synthetic --port 9400 --items 65536 --dim 64
build/Release/recserve_loadgen --port 9400 --qps 200 --n 500
```

## Binaries

| binary | role |
|---|---|
| `recserve_server` | TCP retrieve-rank service |
| `recserve_loadgen` | open-loop client at a fixed arrival rate |
| `recserve_bench` | p99/QPS/RSS/recall in one JSON snapshot; the agent's observation source |
| `recserve_fixture` | build a catalog + HNSW index once and write them to disk |
| `recserve_quality` | recall / NDCG on an interaction CSV |
| `recserve_nearline` | ingest service; `--feature-store mutex\|snapshot`, `--source file\|kafka`, `--replay` |
| `recserve_rankd` | ranker process with a bounded queue and load-shed |
| `recserve_soak` | leak/RSS soak (`--seconds 86400` for 24 h) |
| `recserve_eval` | recommendation quality on real embeddings, through the request path |
| `recserve_shard` | sharded scatter-gather retrieval and tail amplification |
| `recserve_diagnose` | bottleneck classifier |
| `recserve_tests` | 24 core tests; additional GPU, batching, artifact, bundle and real-socket suites run through CTest/Python |

## Scripts

| script | what it produces |
|---|---|
| `measure.py` | the whole campaign → `results/measured.json` + `COST.md` |
| `validate_hnsw.py` | RecServe vs hnswlib on identical data |
| `nearline_ab.py` | mutex vs RCU snapshot under live ingest |
| `skew.py` | online/offline feature disagreement vs publish interval |
| `agent.py` | the closed loop; `--compare`, `--validate-pr` |
| `cost_model.py` | hosts and dollars from measured QPS and bytes |
| `offline_ctr.py` | batch CTR reference the Flink job must match |
| `prep_movielens.py` | download MovieLens, fit ALS, export a real catalog |
| `eval_baselines.py` | random/popularity baselines + C++ vs numpy cross-check |
| `compare_flink.py` | Flink output must equal the Python reference exactly |
| `perf_ci.py` | p99 regression gate |

## Kafka and Flink

Events are 17 bytes: `event_time_ms, user_id, item_id, type` — identical whether
they come from a file, a socket or librdkafka, so CI exercises the whole
nearline path without a broker.

- **Consumer**: [`src/kafka_source.cpp`](src/kafka_source.cpp) is a real
  `RdKafka::KafkaConsumer` with explicit offset commits and watermark-based lag
  (`rd_kafka_query_watermark_offsets`, not the stats callback, whose watermarks
  go stale for idle partitions). Build with `-DRECSERVE_WITH_RDKAFKA=ON`.
- **Offline**: [`flink/ctr_aggregate.py`](flink/ctr_aggregate.py) is a PyFlink
  job — event-time tumbling windows, bounded-out-of-orderness watermarks, per-item
  CTR to CSV.
- `docker compose up -d` brings up Redpanda plus a Flink jobmanager and
  taskmanager.

**Both are verified in CI, not asserted.** The `kafka-integration` job produces
50,000 records to a Redpanda service container and requires the C++ consumer to
drain all 50,000, finish below 100 records of lag, and report a freshness p99
strictly greater than zero. The `flink-job` job runs the actual Flink job on a
real Flink runtime and requires its output to equal
[`scripts/offline_ctr.py`](scripts/offline_ctr.py) exactly — the producer dumps
the bytes it sent, so the two sides compare implementations rather than
datasets. Latest run: **1,822 items agree exactly, 20,000 events accounted
for**.

Two defects the first green CI run exposed, both now fixed and both guarded by
those assertions: `query_watermark_offsets` was a synchronous broker round-trip
inside the ingest loop, holding throughput to 1,116 events/s against a 25,000
events/s producer and reporting the resulting backlog as consumer lag; and
freshness read exactly 0.0 ms because the consumer stamped `steady_clock` (time
since boot) while the producer stamped Unix epoch. A broken metric looked like a
perfect one.

`apache-flink` does not build on Windows/ARM64, so the Flink job runs in CI and
docker rather than on the host the latency numbers came from.

## Workloads

- **Scale track**: synthetic embeddings, clustered by default (`--clusters N`).
  i.i.d. Gaussian directions are uniform on the sphere with no intrinsic
  structure — the documented worst case for graph ANN. Trained embedding tables
  are clustered, and it moves recall@10 at ef=64 from 0.71 to 0.80 on 16K items.
- **Quality track**: temporal split on an interaction CSV. Official
  [KuaiRec](https://kuairec.com/) (CC BY-SA 4.0); convert with
  `python scripts/prep_kuairec.py`. Do not use it as the QPS catalog — it is too
  small for ANN to matter.
- **Event stream**: Zipf item popularity, not uniform. Uniform made every
  per-item counter equally warm and hid that the skew budget is spent on the
  cold tail.

## CI

The workflow expands to 16 jobs; all passed at checkpoint `7e0eb63`:

| job | what it proves |
|---|---|
| `build-and-test` | builds and tests under none/ASan/UBSan/TSan |
| `index-correctness` | RecServe HNSW tracks hnswlib on identical data |
| `perf-gate` | interleaved base/candidate measurements, negative gates and a credential-free agent loop |
| `avx2-and-portability` | Windows/Linux builds with explicit AVX2 ON/OFF |
| `cuda-compile-only` | CUDA toolchain compiles all targets; no GPU execution claim |
| `portable-container` | build and execute tests inside the CPU image |
| `offline-reference` | C++ online aggregate == Python batch aggregate |
| `kafka-integration` | real broker, real librdkafka consumer, lag and freshness are real |
| `flink-job` | the Flink job runs and its output equals the reference exactly |
| `quality-real-data` | ALS on MovieLens; C++ reproduces numpy; HNSW recall > 0.95 |

## Working on this

[`AGENTS.md`](AGENTS.md) carries the handoff notes: current verified state, how
to set up on a new machine (the ~926 MB of fixtures are gitignored and must be
regenerated), the next task, and the traps worth knowing before changing
anything.

## What this does not do

Stated plainly, because the gap between "written" and "verified" is where
projects like this usually mislead.

- **GPU release validation is incomplete.** Actual CUDA correctness and socket
  batching pass, but Compute Sanitizer needs an approved Windows debugger setting.
  GPU throughput crossover is not a proof of online latency or cloud economics.
- **No durable end-to-end feature service.** The Kafka/nearline demonstration is
  not yet a recoverable event-to-response pipeline wired into the deployed server.
- **No public production API.** TCP IDs are internal rows, and serving-time
  eligibility filters, auth/TLS, cold start and real-user outcome logging remain.
- **Shards are threads, not hosts.** No network, no separate failure domain, no
  cross-host variance. The tail amplification numbers are a lower bound.
- **No cross-region replication.** One process, one machine.
- **The ranker is four hand-set weights.** Item embeddings are trained (ALS);
  the ranking model on top of them is not. This serves a model, it does not
  learn one.
- **Historical cost estimates are not quotes.** They are compute-only models
  using their recorded instance assumptions, not verified current all-in costs.
  The RTX experiment supplies no measured cloud-GPU dollar-per-query claim.

Everything else in this README is produced by `python scripts/measure.py` on the
host named at the top, or by a CI job you can open and read.

## License

MIT. See [LICENSE](LICENSE).
