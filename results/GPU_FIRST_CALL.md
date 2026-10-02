# GPU first-call timeline: 2026-10-02

Nsight now localizes the long first request to a host-side cuBLAS heuristic
range before the first device kernel. The range contains multiple library-load
calls. The trace does not separate CPU execution from descheduling inside that
range, so it does not prove that algorithm computation alone consumed the time.

This is the legacy RSV1 CUDA experiment on the workstation's RTX 3060 under WSL.
The authenticated RSV2 pilot remains CPU-only. It is not a capacity, tail-latency,
cloud, sanitizer or production-readiness result.

## Executed experiment

Two sequential campaigns each interleave three traced and three untraced fresh
server processes (fixed shuffle seed 17). Each process receives the same query
three times: first call, after 50 ms, and after two seconds idle. Batch size is
one, retrieval count 64, returned count 10, and the supplied request deadline is
one second. All 18 requests per campaign returned the expected response shape,
stable IDs within the process, and no GPU fallback. The catalog has 6,298 items
and 32 dimensions; it is not the million-item scale board.

The first campaign established the finding. The second reran the complete
experiment after adding correlation-ID matching and clock-interval rejection
to the parser. Both campaigns, including their variability, are retained in
[gpu-first-call-2026-10-02.json](gpu-first-call-2026-10-02.json).

| Measurement across three processes | Initial capture | Verified parser capture |
|---|---:|---:|
| Untraced first-request client time | 82.861–111.311 ms | 61.151–201.640 ms |
| Traced first-request client time | 82.436–105.716 ms | 110.889–153.527 ms |
| First cuBLAS heuristic range | 78.691–102.329 ms | 104.961–148.517 ms |
| Sum of four first-request device kernel durations | 0.048168–0.048857 ms | 0.047600–0.048594 ms |
| Untraced client time after two seconds idle | 0.841–1.082 ms | 0.956–1.584 ms |

The verified capture's repeated heuristic ranges take 0.027–0.038 ms and contain
no further library-load calls. A repeated traced request still took 5.029 ms at
the client; an untraced repeat took 4.191 ms. These outliers and the wide cold
range matter: three repetitions and a shared desktop do not establish a
five-millisecond latency guarantee.

CUDA API durations, cuBLAS ranges, GPU event intervals and kernel durations
overlap. Do not add them. The parser joins kernels to request API correlation
IDs, requires exactly three completed request intervals, rejects kernels outside
their synchronization interval, and fails on empty or missing trace data.
Four negative/correlation tests run in hosted CPU test discovery. They validate
the parser, not GPU execution.

## Interpretation and next experiment

The evidence supports moving a deliberate representative first call before
GPU readiness as the next hypothesis to test. NVIDIA documents host-side
heuristic work and caching in [cuBLAS heuristics](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache).
The trace shows library loads within this work, but cannot assign all remaining
time to a specific initialization mechanism.

Implement warmup only as an explicit policy on the actual GPU worker; cover
supported batch shapes, fail readiness if warmup fails, and measure added startup
time. Compare cold first requests with and without that policy, then test longer
idle gaps, restarts and bursts. Do not infer that warming one query shape covers
all batch sizes or that a two-second gap represents a cold GPU.

CPU HNSW remains the default. Compute Sanitizer is still a separate unpassed
owner-approval gate. No GPU debugger, global ASLR, driver or power policy changed.

## Reproduce and evidence

Build the existing CUDA configuration, then run its tests before profiling:

```bash
cmake --build build-gpu -j 4
ctest --test-dir build-gpu --output-on-failure
python scripts/profile_gpu.py --nsys /path/without/spaces/nsys \
  --output .cache/nsight/new-campaign/report.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

This run used Nsight Systems 2026.1.3.425, package SHA-256 recorded in the JSON,
extracted into an owned temporary directory without a system installation.
An initial cache path containing spaces caused LD_PRELOAD warnings; that capture
was discarded from these results. Both published campaigns use the clean path.
Tracing enables CUDA and cuBLAS only; CPU sampling and context-switch tracing
are disabled. This restricts attribution and avoids system permission changes.

Raw .nsys-rep, SQLite, workload responses and server logs remain under
.cache/nsight/campaign-1 and campaign-2. The published JSON records trace hashes,
commands, source checkpoints, binary/manifest hashes and individual API/kernel
events. The verified script refuses to overwrite an existing report.

Validation: all eight CUDA CTest entries and all 32 Python regression tests
passed locally. Monitoring checkpoint 07a2edb separately passed all 20 hosted
jobs (37057564356 and 37057564318). Hosted runners did not execute this GPU
capture; the existing manual owner GPU workflow and sanitizer requirement remain.
