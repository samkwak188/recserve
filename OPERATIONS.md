# Local execution and production boundaries

## Local feedback pilot

The supported feedback workflow is documented in `docs/PILOT.md`.
PowerShell: `.\scripts\Check-Pilot.ps1`; WSL: `bash scripts/check_pilot.sh`.
It adds a separate loopback policy API with SQLite feedback recovery and
eligibility. It does not change the raw C++ TCP contract described below or
integrate the Kafka/RCU experiments. No public authentication is provided.
GPU stage diagnostics and their limits are in `results/GPU_INVESTIGATION.md`.

## Verified environment

Windows host: Ryzen 5 5600X, 12 logical CPUs, RTX 3060 12 GiB, driver 591.86.
Execution: existing Ubuntu 24.04 WSL, GCC 13.3, CMake 3.28, CUDA compiler 13.1.115.
AVX2/FMA is an explicit build requirement when enabled, not runtime dispatch.

`scripts/run_logged.py` retains full command output in `.cache/logs/<name>.log`
and a JSON status/exit-code/duration receipt. Commands run without a shell;
POSIX timeouts kill the command's process group. Windows timeouts kill the parent.
Do not confuse an older completed receipt with an ongoing run from another date.

## Reproduce locally

Run the following inside Ubuntu WSL from the repository root. Bootstrap needs
root inside WSL, not a replacement NVIDIA driver. Use the installed Windows GPU
driver; do not install a Linux display driver into WSL.

```bash
sudo bash scripts/bootstrap_wsl.sh
bash scripts/setup_python.sh
bash scripts/check_cpu.sh
RECSERVE_TSAN_NO_ASLR=1 bash scripts/check_foundation.sh
bash scripts/check_real_data.sh
bash scripts/check_gpu.sh
bash scripts/check_integrated.sh
bash scripts/final_campaign.sh
```

Python tooling uses Python 3.12+ and `.cache/venv`. `requirements.txt` pins direct
dependencies; it is not a transitive lockfile. Data preparation uses CPU ALS and
explicitly disables the optional implicit CUDA extension. The C++ CUDA backend
is independent of that Python package.

`check_gpu.sh` currently returns failure at Compute Sanitizer on this workstation
because the Windows GPU debugger interface is disabled. Its earlier build/tests
pass. Do not delete this failing gate or relabel it green. NVIDIA documents the
administrator-controlled `EnableInterface` setting under
`HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger`; enabling it requires owner approval.
See [Compute Sanitizer setup](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
and [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/).

ThreadSanitizer's WSL workaround is scoped to the test process (`setarch -R`). No
system-wide ASLR setting was changed. Ordinary production runs retain ASLR.

## Serve the trained model

```bash
python3 scripts/bundle.py verify --manifest data/wsl_small_bundle.json
python3 scripts/bundle.py serve --manifest data/wsl_small_bundle.json \
  --binary build/recserve_server --backend hnsw
```

Use `build-gpu/recserve_server --backend cuda` through the same bundle launcher
for GPU serving. CUDA absence fails startup; a runtime GPU failure trips exact
CPU fallback, visible in metrics. Restart after repairing the GPU to reset that
circuit. Automatic GPU retries are deliberately not implemented.

The server requires real catalog/query paths unless `--synthetic` is explicitly
selected. Defaults: bind 127.0.0.1:9400; health/metrics 127.0.0.1:9401; 8 connection
workers; 64 pending connections; 250 ms absolute frame/write I/O limits. Additional
kernel listen backlog is 128. Connection overload closes the unparsed connection;
it cannot attach a LoadShed response to an unknown request ID. Request expiration
includes GPU queueing/batching; computation is not preemptible.

GPU defaults: max batch 8, batch formation wait <=100 us, 64 scheduler slots;
max retrieve-k 512. Configure the executable with `--batch`, `--batch-wait-us`,
`--workers`, `--queue` and `--io-ms` for experiments. Do not copy a preformed-batch
throughput optimum into an online latency-sensitive service without testing it.

HTTP admin endpoints: `/healthz`, `/readyz`, `/metrics`. Readiness currently means
the process loaded its model and opened the listener; it does not certify durable
feature freshness. Metrics expose connection/request/error counts, RSS and GPU
batch/fallback state. Latency distributions are presently measured by the load
client, not a full server-side histogram dashboard.
Clean EOF disconnects have a separate counter; the initial `7e0eb63` capacity
artifact counted them in its server I/O counter. Use the client-side failure
accounting in that artifact, not the old server I/O total, to assess failures.
Ranking scratch reuse (`--no-arena` disables it) is not a zero-allocation promise:
retrieval and response vectors may still allocate.

TCP protocol is the existing RSV1 fixed-width little-endian protocol. User IDs
are dense query rows, not raw MovieLens IDs. The bundle preserves both identity
maps; the public-ID gateway and cold-start policy are still to be implemented.
No TLS/authentication is built into this internal transport. Do not expose it
publicly. The old rankd and shard tools are experiments, not hardened endpoints.

## Container proof

Prepare the bundle first, then with Docker Desktop running:

```bash
docker build -t recserve:local .
python3 scripts/check_container.py
docker compose -f compose.serve.yml up -d --no-build
docker compose -f compose.serve.yml down
```

The smoke script uses an isolated `recserve-validation` project and tears down
its own container/network. It never deletes model data or volumes. The serving
image runs UID/GID 10001 with read-only model mount, read-only root, dropped
capabilities, no-new-privileges, 1 GiB RAM, 128 PIDs, and a loopback-published port.
The base image uses an explicit Ubuntu release tag, not a digest-locked supply
chain. GPU runtime images and vulnerability/signature gates remain release work.

## Operational response and rollback

- Elevated rejection/deadline counts: lower offered load, inspect scheduler lag
  in the client report, then compare CPU and GPU under the same quality contract.
  Increasing queue capacity usually hides overload while worsening latency.
- GPU health 0/fallback rising: preserve the error context and device diagnostics,
  verify CPU capacity, then restart only after repairing the device/runtime.
- Failed bundle verification: refuse startup. Restore the previous complete,
  verified bundle in a separate directory and restart against it. Do not edit
  individual live catalog/index/query files in place.
- Stale features: raw TCP has no production feature bridge. The local pilot reads
  its transactional projection directly and fences stale response publication;
  database unavailability returns an error. This is not R3's Kafka recovery or
  an asynchronous freshness guarantee. Complete those gates before live ingestion.
- Shutdown: SIGTERM/SIGINT stops admission, closes queued connections and lets
  active work finish/expire. Use a supervisor/container grace period and verify
  this under the largest permitted catalog; a stuck hardware operation is not
  bounded by a C++ cancellation mechanism.

These are local procedures, not completed multi-replica rollback/restore drills.
Release gates and remaining implementation details are in `EXECUTION_PLAN.md`.
