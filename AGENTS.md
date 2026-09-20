# Agent handoff

## Continuation checkpoint: 2026-09-20

Read `EXECUTION_PLAN.md`, `OPERATIONS.md` and `results/EXECUTION_REPORT.md` first.
Runtime commit `7e0eb63` adds actual RTX 3060 CUDA exact retrieval, GPU batching,
bounded serving, integrity-checked model bundles, safety repairs and CPU container
validation. All 16 expanded hosted jobs passed in run 35541070732. Local CUDA
functional tests pass; Compute Sanitizer is blocked by the Windows debugger
interface and requires owner approval, not a skipped-green gate. The manually
dispatched GPU workflow has no registered persistent owner runner.

Follow-up `4651d0d` fixes clean-disconnect metrics and constrains CI dependencies;
all 16 jobs passed again (run 35542148547). Measurement reports identify their
original runtime checkpoint rather than silently relabeling older measurements.

Use existing Ubuntu WSL on this Windows/Ryzen/RTX workstation. Commands and full
logs live in `scripts/` and `.cache/logs/`. CPU sanitizer matrix passes; WSL TSan
uses `RECSERVE_TSAN_NO_ASLR=1` for a process-only workaround. Never change global
ASLR or Windows GPU debugger settings without approval.

The original handoff below is historical (2026-09-17 ARM host), retained for its
design context and traps. Its GPU-not-done statement, test count, and old next
task are superseded. Preserve its ARM measurement artifacts. Use separate WSL
results and never equate preformed-batch QPS with online service capacity.

Next work is the gated R1-R5 roadmap, especially a chosen pilot and durable
event-to-response/eligibility parity. Do not claim production readiness or online
user lift from the offline MovieLens experiment.

---

Working notes for whoever picks this repo up next, human or agent. Written at
the end of a session that took RecServe from a benchmarked prototype to a
project where every claim has a measurement or a CI job behind it. The next
step needs an NVIDIA GPU, which the previous host did not have.

Read this before changing anything. The traps section exists because each entry
cost real time to find.

This file is the single source of handoff context for this repo — there is no
separate per-tool instructions file. If your agent harness auto-loads a
different filename, point it here rather than duplicating this content.

---

## 1. Ground rules

**Identity.** Commit and push as `Sam Kwak <ckwak7@wisc.edu>`. This is the
owner's personal repo (`github.com/samkwak188/recserve`).

**No AI attribution anywhere.** No `Co-Authored-By` trailers, no "generated
with", no mention of any assistant in commit messages, code comments, or docs.
The history is clean; keep it that way. (`--provider claude|gemini` in
`scripts/agent.py` is an API backend identifier, not attribution.)

**Commit per phase, not per session.** One coherent change per commit, pushed
as it lands. Commit messages here are long on purpose: they record *what was
measured and why a change was made*, not just what changed. Match that.

**Nothing ships unexecuted.** The single biggest lesson of the last session:
code that has never run is worth less than the sentence admitting it hasn't.
A Flink job sat in the tree for hours looking complete and could never have
executed. If you add a capability, add the CI job that proves it runs.

---

## 2. What the project is

A C++20 two-stage recommendation serving engine: retrieve a few hundred
candidates from an HNSW index, rank them with a linear scorer over nearline
features, return top-K. Around it: a Kafka ingest path, a Flink batch
aggregate, a lock-free feature snapshot, a sharded scatter-gather retriever, a
closed-loop tuning agent, and a capacity/cost model.

Read `README.md` for the measured results and `COST.md` for the full board.
Both are generated or hand-checked against `results/*.json`.

---

## 3. State as of this handoff

Seven CI jobs, all green, four consecutive green runs. 22 unit tests. ~7,900
lines of C++/Python. Clean working tree.

| Capability | Status | Proven by |
|---|---|---|
| HNSW index | verified against hnswlib, max recall delta 0.0082 | `index-correctness` CI job |
| SIMD/int8 kernels | int8 fastest, 5.3x over scalar on 1M items | `scripts/measure.py` kernels stage |
| Kafka consumer | 50,000/50,000 drained, lag 0, real freshness | `kafka-integration` CI job |
| Flink batch job | runs on real Flink, output == Python reference exactly | `flink-job` CI job |
| RCU feature snapshot | p99 1.6-2.7x better than mutex under live ingest | `scripts/nearline_ab.py` |
| Real-data quality | ALS +43.0% over most-popular; C++ == numpy to 0.00000 | `quality-real-data` CI job |
| Sharding | tail amplification 1.19x -> 2.25x (1 -> 8 shards) | `scripts/measure.py` shard stage |
| Closed-loop agent | +64.9% from a misconfigured start, confirmed | `scripts/agent.py --compare` |
| GPU scoring | **NOT DONE** — no CUDA device was available | — |

---

## 4. Setting up on the new machine

A clone gets source and `results/`. It does **not** get any fixture: `data/*.bin`
and the generated CSVs are gitignored (~926 MB locally). Regenerate them.

```bash
git clone https://github.com/samkwak188/recserve && cd recserve
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure     # must be 22/22
```

Python side (the measurement and data tooling):

```bash
pip install numpy scipy implicit pandas hnswlib kafka-python pydantic certifi pyyaml
```

Regenerate fixtures — none of this is optional if you want to reproduce the board:

```bash
# synthetic scale track: 64k (fast) and 1M (the published board, ~100 s build)
./build/recserve_fixture --items 65536  --dim 64 --clusters 1024 --queries 4096 \
    --out-catalog data/catalog_64k.bin   --out-index data/index_64k.bin
./build/recserve_fixture --items 1000000 --dim 64 --clusters 4096 --queries 4096 \
    --out-catalog data/catalog_1000k.bin --out-index data/index_1000k.bin

# real data: downloads ml-25m (262 MB, cached in data/movielens/), fits ALS
python scripts/prep_movielens.py --dataset 25m --factors 64 --iterations 20
./build/recserve_fixture --in-catalog data/ml25m_catalog.bin \
    --out-index data/ml25m_index.bin --m 16 --ef-construction 200 --build-threads 1

# nearline event log
./build/recserve_nearline --write-log 200000 --items 16384 --out data/events.bin
```

Then the full campaign (regenerates `results/measured.json` and `COST.md`):

```bash
python scripts/measure.py            # all 11 stages; takes a while
python scripts/measure.py --quick    # cheap stages only
python scripts/measure.py --board-only   # reformat COST.md, no re-measuring
```

`measure.py` **merges** into existing results, so running one `--stages X` does
not wipe the others. It did wipe them before; do not reintroduce that.

**Note on ISA.** The previous host was Windows on ARM64 (Snapdragon,
`neon+sdot`). An x86 machine takes the AVX2 paths in `include/recserve/simd.hpp`
instead. Those paths compile and are exercised by Linux CI, but the published
kernel numbers in `README.md`/`COST.md` were measured on ARM64. **Re-run
`scripts/measure.py` on the new host and regenerate the board** — do not mix
numbers from two ISAs in one table. Update the host line at the top of
`COST.md` (it is generated, so this happens automatically).

---

## 5. The next task: GPU scoring

This is why the repo is moving. The JD this project targets names "GPU compute
efficiency" explicitly, and it is the last unaddressed gap.

### The question to answer

**At what candidate count does GPU scoring beat CPU, and what is the $/query on
each side?** Not "does the GPU work" — a crossover point with a cost figure.

### Where it fits

Two distinct opportunities; the first is the better one:

1. **Exact scan / brute-force retrieval.** Scoring a query against the whole
   catalog is a GEMV (one query) or GEMM (a batch). This is what GPUs are for,
   and on 1M items the CPU takes ~6 ms (simd) / ~5.3 ms (int8) — see the
   "Kernels, exact scan (1,000,000 items)" table in `COST.md`. A batched cuBLAS
   version should be far faster, and the interesting result is the **batch size
   at which the GPU's fixed overhead is amortised**, plus whether GPU brute
   force beats *CPU HNSW* (which is ~100 µs — a much harder bar than CPU brute
   force, and the honest comparison).
2. **The ranking stage.** Smaller win; retrieval dominates the profile. Do it
   second if at all.

Do **not** attempt GPU HNSW graph traversal. It is a research problem
(irregular memory access, poor warp utilisation) and not what this project is
demonstrating.

### Suggested shape

- `src/gpu_scorer.cu` + `include/recserve/gpu.hpp`, behind
  `-DRECSERVE_WITH_CUDA=ON`, defaulting OFF, with `make_gpu_scorer()` returning
  null and a message when unavailable — mirror how
  `src/kafka_source.cpp` handles a missing librdkafka. Every other target must
  still build without CUDA.
- Add `Kernel::Cuda` to the enum in `include/recserve/types.hpp` and a
  `score_range` path in `Catalog` (`include/recserve/scorer.hpp`). Keep the host
  buffers as they are; upload the catalog once at load, not per query.
- Extend `apps/bench.cpp` with `--kernel cuda` so the existing JSON snapshot,
  the agent's observe step, and `perf_ci.py` all work unchanged.
- Add a `gpu` stage to `scripts/measure.py` sweeping batch size
  (1, 8, 32, 128, 512, 2048) against the CPU kernels at the same catalog size.
- Extend `cost/instances.json` with GPU instances (g5.xlarge, g6.xlarge) and
  teach `scripts/cost_model.py` that a GPU host has a different
  throughput-per-dollar shape. Re-check the "bound by" logic: a GPU fleet is
  usually memory-capacity-bound long before it is compute-bound.

### Verification bar

Matching everything else here:
- a unit test asserting the CUDA kernel returns the **same top-k** as
  `Kernel::Simd` on the same catalog (tolerance on scores, exact on ids for
  well-separated cases) — see `test_blocked_matches_simd` for the pattern;
- a measured crossover point, not an assertion that GPUs are fast;
- CI: GitHub's hosted runners have **no GPU**. Either add a self-hosted runner
  with the GPU, or gate the CUDA job on `runs-on: [self-hosted, gpu]` and state
  plainly in `README.md` that it runs on the owner's machine and not in hosted
  CI. Do not let the job silently skip and look green.

When it works, delete the GPU bullet from the "What this does not do" section
of `README.md`. If it does not work, leave that section accurate.

---

## 6. Traps (each of these cost real time)

**Measurement**

- **Never rebuild the index inside a measurement loop.** The parallel HNSW build
  is order-dependent: recall@10 spread is 0.0172 over five 12-thread builds and
  0.0000 single-threaded. Load a snapshot. `--build-threads 1` when you need
  reproducibility.
- **Within-process trial variance is not between-process variance.** Trial CV
  inside one benchmark process badly understates run-to-run drift (5.7% at
  n=600 on the old host). `scripts/agent.py` calibrates σ by running the same
  config five times before it will accept anything. Keep that.
- **Re-measure the incumbent beside the candidate.** Comparing a number from
  ten minutes ago to one from now compares thermal states.
- **A win must replicate.** Twelve comparisons at 2σ produce false positives by
  construction; one slipped through before replication was added.
- **`ef` below `retrieve_k` is a no-op** — the engine searches with
  `max(ef, retrieve_k)`, so an ef sweep at fixed retrieve_k looks flat below it.

**Correctness**

- **Working directory.** `ctest` runs from the build tree; running the binary by
  hand usually happens from the repo root. Tests write scratch files to
  `temp_directory_path()/recserve_tests` for exactly this reason. Every CI job
  was red for hours because of a `data/...` path that existed in one case.
- **Clock domains.** `now_us()`/`now_ns()` are `steady_clock` and are for
  *durations only*. Anything crossing a process boundary uses
  `now_ms_epoch()` (`system_clock`). Mixing them made Kafka freshness read
  exactly 0.0 ms and look perfect.
- **Reserved words in Flink SQL.** `views` is one. Quote every identifier with
  backticks.
- **`recserve_bench` rejects unknown flags.** It used to ignore them, which
  silently reported numbers for the wrong configuration.

**Things that were broken and are now fixed — do not regress them**

- `FlatUserMap::get()` looped forever when more distinct user ids arrived than
  were reserved. It grows now; `test_flat_user_map_grows_past_reservation`
  guards it.
- `query_watermark_offsets` was a synchronous broker round-trip in the Kafka
  ingest hot loop, capping throughput at 1,116 eps against a 25,000 eps
  producer. It refreshes on a 500 ms interval now.
- The RCU concurrency test depended on thread-start latency and passed or failed
  on scheduling. Readers use do-while and the writer waits on a ready counter.

---

## 7. Map of the repo

| Path | What |
|---|---|
| `include/recserve/simd.hpp` | SIMD kernels, runtime ISA detection (NEON SDOT / AVX2) |
| `include/recserve/scorer.hpp` | `Catalog`, layouts (AoS / SoA / AoSoA / int8), disk format |
| `include/recserve/index.hpp` | HNSW (Malkov & Yashunin algorithms 1, 2, 4, 5) |
| `include/recserve/engine.hpp` | request path: features -> retrieve -> rank |
| `include/recserve/feature_snapshot.hpp` | RCU feature store, explicit grace period |
| `include/recserve/event_source.hpp` | file vs Kafka behind one 17-byte record |
| `src/kafka_source.cpp` | librdkafka consumer, watermark lag |
| `apps/bench.cpp` | the measurement harness; also the agent's observe step |
| `apps/eval.cpp` | recommendation quality on real embeddings |
| `apps/shard.cpp` | scatter-gather + tail amplification |
| `apps/nearline.cpp` | ingest service, `--feature-store mutex\|snapshot`, `--replay` |
| `flink/ctr_aggregate.py` | the Flink batch job (Table API, raw format, UDFs) |
| `scripts/measure.py` | the whole campaign -> `results/` + `COST.md` |
| `scripts/agent.py` | closed loop: observe/diagnose/plan/execute/verify/improve |
| `scripts/validate_hnsw.py` | RecServe vs hnswlib on identical data |
| `scripts/prep_movielens.py` | download, ALS, export a real catalog |
| `scripts/eval_baselines.py` | baselines + C++/numpy cross-check |
| `scripts/cost_model.py` | measured throughput and bytes -> hosts and dollars |

---

## 8. If you only do one thing first

Run `ctest --test-dir build -C Release` and `python scripts/measure.py --quick`.
If both are clean, the port is sound and you can start on the GPU work. If the
kernel numbers look nothing like `COST.md`, that is expected on a different ISA
— regenerate the board rather than reconciling two hosts in one table.
