# GPU readiness warmup: 2026-10-02

Explicit GPU warmup removed the first-request deadline misses in this local
experiment: 6/6 cold first requests missed 5 ms; 0/6 warmed first requests missed.
Warmup did not qualify steady serving or idle recovery: one warmed request after
15 seconds idle missed, and bursts still failed. CPU remains the default.

## Change and executed contract

Runtime checkpoint 5890b27 adds --gpu-warmup, default off, to the C++ server and
verified bundle launcher. The GPU worker executes every batch count from one
through the configured maximum before either socket opens. It uses the largest
legal top-k and the loaded queries. Errors abort startup and join the worker;
startup does not silently fall back to CPU.

Separate warmup metrics keep these calls out of request counters and first-batch
timings. CPU tests block a scorer to prove startup waits, inject failure, check
partial batch shapes and verify that warmup and serving use the same worker.
Real CUDA service tests exercise warmup both on and off. Separate launch checks
execute the verified bundle launcher and all 64 permitted warmup batch shapes.

See [operator contract](../OPERATIONS.md#explicit-gpu-readiness-warmup).
A supervisor must impose a startup deadline; in-process code cannot safely
preempt a hung device call.

## Measurement correction found during validation

The initial campaign produced successful latency percentiles above its 5 ms
deadline. Its asyncio timeout callback could run late when the event loop was
delayed. Checkpoint 36e8663 adds an explicit completion-versus-intended-arrival
check. Late responses count as client deadline failures and are excluded from
successful latency samples.

The original harness fails both deterministic late-completion tests; the fix
passes them on Linux and native Windows. Historical open-loop failure counts
may therefore be optimistic about strict deadlines. They retain their original
checkpoints, with a caveat; they are not silently relabeled.

The initial campaign is retained as rejected evidence inside
[gpu-warmup-2026-10-02.json](gpu-warmup-2026-10-02.json). Only the rerun with the
corrected harness supports the table below. It also asserts that every reported
successful p99 is below 5 ms and all attempts are accounted for.

## Corrected experiment

Twelve independent processes: three repetitions of warmup on/off at maximum
batch sizes 1 and 8, interleaved with shuffle seed 31. Every process uses the same
verified MovieLens-small catalog (6,298 items, 32 dimensions), RTX 3060, and
100 microsecond batch wait. No profiler runs during this campaign.

Each process receives one first request, two seconds at 100 offered requests/s,
15 seconds idle, one post-idle request, then two seconds at 1,000 requests/s.
All client deadlines are 5 ms from intended arrival, including queueing and
transport. The client uses 32 workers and a 256-job queue. All 26,424 attempts,
including failures, remain in the JSON. No GPU fallback occurred.

| Maximum batch | Warmup | First misses | Low-rate misses | Post-idle misses | Burst misses |
|---|---|---:|---:|---:|---:|
| 1 | Off | 3/3 | 23/600 | 0/3 | 364/6,000 |
| 1 | On | 0/3 | 0/600 | 0/3 | 426/6,000 |
| 8 | Off | 3/3 | 19/600 | 0/3 | 108/6,000 |
| 8 | On | 0/3 | 0/600 | 1/3 | 147/6,000 |

Cold first GPU-batch wall time ranges from 58.861 to 142.731 ms.
With warmup it ranges from 0.263 to 0.668 ms; the first client response takes
0.799 to 1.430 ms. These are six observations per setting, not percentile
guarantees or a comparison against CPU HNSW.

Warmup itself takes 58.978–270.842 ms. Total observed start-to-readiness ranges
from 304.061–368.279 ms without warmup and 354.291–1,217.346 ms with it.
Readiness is polled every 50 ms; total startup includes artifact and CUDA
initialization. Do not subtract cohort endpoints to estimate warmup overhead:
the direct warmup metric is the relevant measurement.

The warmed cohort has more total misses across these short workloads
(574/13,212 versus 520/13,212). Thus this supports a narrow startup fix, not
a general failure-rate, throughput or burst-performance improvement. A shared
desktop, client scheduling variation and three processes per cell limit causal
claims about the later workload stages. Successful latency samples alone would
hide those misses; they must be read alongside counts.

## Validation and remaining work

All five local configurations pass: CPU Release, CUDA Release (9 CTest entries),
AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer. TSan uses only
the existing process-scoped ASLR workaround. All 36 Python discovery tests pass.
Hosted CPU/portability/container validation passed all 20 jobs for runtime
5890b27 and harness correction 36e8663. Hosted runners did not execute CUDA.

Compute Sanitizer remains an unpassed owner-approval gate. No debugger, driver,
global ASLR or power setting changed. RSV2 authenticated serving is still
CPU-only; this experiment concerns the legacy RSV1 retrieval path.

Next: isolate post-idle and burst misses using per-phase queue/device/client
evidence, compare quality-matched CPU and GPU paths under the corrected harness,
and extend run duration and catalog scale. Do not promote GPU by default from
this small startup experiment. Cloud deployment, independent external alerting
and fresh-host restore remain separate release gates.

Reproduce after building/testing the CUDA configuration:

```bash
python scripts/measure_gpu_warmup.py \
  --out .cache/gpu-warmup/new-campaign/report.json
```

The script refuses to overwrite an existing report. Source, binary, manifest,
harness hashes, exact commands, per-process metrics and server-log hashes are
recorded. Full local logs are in .cache/logs and .cache/gpu-warmup.
