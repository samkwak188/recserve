# Execution report: RTX/WSL continuation

Runtime checkpoint: `7e0eb63`, 2026-09-20. This is a verified local prototype, **not a production-ready pipeline**.

Full acceptance gates: [execution plan](../EXECUTION_PLAN.md). Reproduction and operational limits: [operations](../OPERATIONS.md).

## Verification

- All 16 hosted jobs passed: [runtime CI](https://github.com/samkwak188/recserve/actions/runs/35541070732).
- Telemetry/dependency follow-up `4651d0d` also passed all 16 jobs: [follow-up CI](https://github.com/samkwak188/recserve/actions/runs/35542148547).
- CPU-only/AVX2, ASan, UBSan and TSan pass; WSL TSan used process-only ASLR disabling.
- Six local CUDA-build CTest suites pass, including actual RTX correctness and socket microbatching.
- Bundle tampering/identity mapping, malformed frames/artifacts, slow clients, bounded overload, recovery, shutdown, deadline expiry and injected exact CPU fallback are tested.
- Non-root/read-only/loopback CPU container served the trained model; its temporary validation container/network were removed.
- Compute Sanitizer is **blocked**, not passed: Windows GPU debugger interface requires owner/admin approval.
- Matched-ISA paired regression: median p99 ratio **1.022x**, recall delta **-0.0000**; gate passed.

Machine: Ryzen 5 5600X, 12 logical CPUs, RTX 3060 12 GiB; Ubuntu 24.04 WSL, GCC 13.3, CUDA 13.1.115, driver 591.86.
Detailed command receipts and source/binary/fixture provenance: [validation JSON](execution-validation.json) and [GPU raw measurements](gpu-crossover.json).

## Observed GPU crossover

Top-10 exact retrieval, dim64 synthetic catalog, three independent processes per point, 200 timed preformed batches/process after five warmups. Each backend uses the same 128 independent quality queries.
The CPU exact baseline is single-threaded AVX2 FP32 `simd`, not an all-core CPU/GEMM baseline or a claim of best-possible CPU performance. GPU batch latency excludes queue formation, feature lookup, final ranking and network. These sampled p99 values are descriptive, not a production tail bound.

| Items | Batch | GPU QPS | Speedup vs CPU exact | GPU batch p99 ms |
|---:|---:|---:|---:|---:|
| 4,096 | 1 | 6,449 | 0.15x | 0.309 |
| 4,096 | 8 | 42,264 | 0.99x | 0.385 |
| 4,096 | 128 | 303,312 | 7.13x | 0.594 |
| 16,384 | 1 | 6,167 | 0.56x | 0.287 |
| 16,384 | 8 | 38,348 | 3.50x | 0.351 |
| 16,384 | 128 | 133,746 | 12.19x | 1.247 |
| 65,536 | 1 | 4,659 | 1.77x | 0.350 |
| 65,536 | 8 | 20,576 | 7.82x | 0.590 |
| 65,536 | 128 | 42,265 | 16.07x | 3.216 |
| 262,144 | 1 | 1,759 | 4.58x | 0.767 |
| 262,144 | 8 | 6,701 | 17.45x | 1.439 |
| 262,144 | 128 | 10,895 | 28.37x | 11.999 |
| 1,000,000 | 1 | 528 | 5.25x | 2.136 |
| 1,000,000 | 8 | 1,840 | 18.28x | 4.639 |
| 1,000,000 | 128 | 2,875 | 28.56x | 44.914 |

GPU retrieve recall minimum: **1.000000** on these probes. Maximum explicitly allocated device workspace/catalog across this sweep: **630.3 MiB** (not total process/driver VRAM).
At batch1 the observed GPU/CPU-exact crossover lies between 16K and 64K items. At 4K items batch8 is effectively tied (~0.994x), not a demonstrated win. Large batches improve throughput but can violate a 5 ms online latency target.

### CPU ANN must be compared at stated quality

The main sweep uses HNSW ef384. At 1M synthetic items its recall is only 0.5695, so its QPS is not an equal-quality substitute for GPU exact retrieval. A second fixed-fixture sweep reports:

| HNSW ef | Minimum recall@10 | Median QPS, batch1 |
|---:|---:|---:|
| 384 | 0.569531 | 1,512.8 |
| 1024 | 0.820312 | 554.6 |
| 4096 | 0.983594 | 147.0 |
| 16384 | 1.000000 | 40.3 |

This is a coarse ANN grid and a synthetic low-recall stress case, not an optimized equal-quality real-data benchmark. The measured recall1.0 is finite-probe evidence, not a guarantee over every query. [Raw ANN results](ann-high-recall.json).

## Actual network service capacity

Verified MovieLens bundle: 6,298 items, dim32, 579 query rows; 8 server workers; GPU maxbatch8/wait100us; one TCP connection/request; fixed intended arrivals; 5 ms client deadline; 15 seconds per point.
Latency includes client scheduling, queueing and network. All attempted arrivals are accounted for. Reported p99 is for successful requests, so failure rate must be read alongside it. This is a short sweep without an explicit service warmup, not a one-hour soak or proven saturation capacity.

| Backend | Offered QPS | Attempted | Successful p99 ms | Failure % | Provisional SLO met |
|---|---:|---:|---:|---:|---|
| hnsw | 100 | 1500 | 2.507 | 0.000 | yes |
| hnsw | 500 | 7500 | 1.405 | 0.000 | yes |
| hnsw | 1000 | 15000 | 1.476 | 0.000 | yes |
| hnsw | 2000 | 30000 | 1.800 | 0.000 | yes |
| cuda | 100 | 1500 | 4.872 | 6.267 | no |
| cuda | 500 | 7500 | 4.296 | 0.293 | no |
| cuda | 1000 | 15000 | 3.742 | 0.520 | no |
| cuda | 2000 | 30000 | 2.154 | 0.000 | yes |

**Decision:** keep CPU HNSW as the local default. GPU misses at low/moderate arrival rates must be investigated and retested before an automatic routing policy or GPU SLO claim. The sweep identifies symptoms, not their proven cause.
The existing offline evaluation filters seen items; the deployed TCP path does not yet have those filters or fresh durable features. Therefore this is a serving-system test on real embeddings, not proof of production recommendation utility. [Raw capacity results](service-capacity.json).
A post-campaign telemetry correction separates clean EOF disconnects from I/O errors. The old raw server I/O counter includes normal one-request connection closures; the client-side failure rates in this table do not. The correction is covered by CPU/GPU and thread-sanitized socket tests.

## Real-data quality

ALS recall@10 **0.06394**, popularity **0.04549**: **40.6% relative offline lift**. C++ and NumPy recall agree within the recorded precision. CUDA also reproduced the measured held-out recall.
This uses a per-user temporal holdout, not a globally time-isolated train/validation/test design. No confidence interval, new learned reranker, causal engagement lift or real-user experiment is claimed. [Raw quality report](wsl-quality.json).

## Remaining release gates

NVIDIA sanitizer approval; long-duration/failure-drill validation; selected pilot objective; durable transactional feature recovery and event-to-response wiring; serving-time eligibility/cold-start parity; learned and temporally validated ranking; authenticated deployment, backup/rollback, and opt-in user outcomes.
No cloud instance was rented, no public endpoint exposed, and no real-user impact or cloud cost saving inferred. The historical ARM board is unchanged.

The GPU campaign began before the runtime commit was created, so its recorded Git parent is `86bbbc2`; its source and executable hashes identify the measured worktree. That runtime source was subsequently committed as `7e0eb63`. New WSL artifacts do not replace historical ARM measurements.
