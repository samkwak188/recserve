# Low-rate GPU investigation

Single-host WSL experiment; no trading or whole-pilot latency claim.

Each cell uses three independent service processes, 100 offered QPS for five seconds
and a 5 ms client deadline. Warm cells first run 2 seconds at 1,000 QPS.
Cold means a fresh process without request warmup, not a reset GPU or cold OS.
Policy order is interleaved with a fixed seed. Stage instrumentation is enabled.

| Backend / max batch | Warmed | Failed / attempted | Successful p99 range (ms) |
|---|---|---:|---:|
| cuda / 1 | False | 22/1500 | 2.953-2.999 |
| cuda / 1 | True | 0/1500 | 3.207-3.491 |
| cuda / 8 | False | 19/1500 | 3.137-3.218 |
| cuda / 8 | True | 1/1500 | 3.136-3.397 |
| hnsw / 1 | False | 0/1500 | 2.573-2.863 |
| hnsw / 1 | True | 0/1500 | 2.579-2.732 |

## Stage evidence and interpretation

- First CUDA call host wall time: 54.025-78.982 ms.
- First CUDA call summed device-event stages (H2D/compute/D2H): 57.985-81.794 ms.
- The long first call is directly observed. Device-event intervals and host wall
  measurements disagree at startup; they are not a trustworthy additive decomposition.
  Do not subtract them to claim host-only overhead or pure kernel time. Event intervals
  can include gaps between stream submissions. A timeline/clock investigation is needed
  before attributing this delay to module loading, scheduling or a specific kernel.
  [CUDA event timing reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).
- Compare cold/warm failure counts before deciding whether warmup helps; remaining
  warm misses and sample size prevent a production latency guarantee.
- Batch1 versus batch8 at low rate tests batching policy, not GPU throughput capacity.
  Per-batch stage distributions must not be added to per-request p99 values.
- Queue histograms cover dispatched GPU jobs; expiry counters also include jobs
  discarded before dispatch. Never infer zero queue trouble from survivor histograms.
- Keep CPU HNSW as the default. No automatic router, driver setting, power-state
  change or startup policy was enabled from this experiment.

## Reproduce / remaining work

`bash scripts/check_ownership_runtime.sh` runs the local sanitizer/CUDA matrix
before the experiment. `python3 scripts/summarize_gpu_investigation.py` generates
this report. See [raw data](gpu-lowrate-investigation.json) for command parameters,
source/binary/manifest hashes, individual runs and cumulative stage buckets.

Next: profile first-call host/runtime operations, test readiness warmup as an explicit
policy under idle gaps and restarts, perform longer burst/soak tests, then compare
quality-matched optimized multicore/GEMM baselines on realistic larger catalogs.
Compute Sanitizer remains a separate unpassed approval gate.
