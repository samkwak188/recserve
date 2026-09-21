# RecServe

A C++20 recommendation retrieval engine with a recoverable local movie-feedback
pilot. The immediate product question: can feedback produce useful, eligible
recommendations without losing that feedback after a restart?

## Run the complete local check

On the configured Windows/Ubuntu WSL workstation:

```powershell
.\scripts\Check-Pilot.ps1
```

It builds/tests the engine, issues recommendations through a local API, records
a dismissal, checks the changed response, restarts both processes, and verifies
recovery. Commands live in files; full logs stay in `.cache/logs`. The synthetic
fixture needs no download; an existing MovieLens bundle gets a separate real-data
acceptance run. See [pilot setup and API](docs/PILOT.md).

## Supported core versus experiments

| Component | Current boundary |
|---|---|
| C++ retrieval | CPU HNSW/exact kernels; optional tiled CUDA/cuBLAS exact top-K |
| Internal service | Bounded TCP admission, deadlines, CPU fallback, health/metrics |
| Local pilot | Stable external IDs, training-seen/feedback/availability filters, replenishment, explicit cold start |
| Feedback persistence | Single-host SQLite journal, deduplication, generation-fenced serving, process-crash/restart tests |
| Kafka/Flink/RCU demonstrations | Separate experiments; not the pilot's durable feature transport |
| Sharding, tuning, cost board | Experiments, not distributed HA or verified current cloud savings |

## Three deliberate decisions

1. **CPU is the default.** The small real-data service sweep favored CPU
   reliability. GPU throughput on large preformed batches is a different result.
2. **Correctness before infrastructure.** One transaction owns feedback and its
   projection. No claim of distributed exactly-once processing is needed.
3. **Measure the actual contract.** Issued responses are not impressions;
   display acknowledgements are explicit. Quality, latency and failed requests
   must be evaluated together.

## Evidence and limitations

- [Latest implementation and verification report](results/PILOT_EXECUTION_REPORT.md).
- [GPU/CPU and service measurements](results/EXECUTION_REPORT.md): hardware,
  batch size, quality and timing boundaries accompany the numbers.
- [Pilot contract and recovery checks](docs/PILOT.md).
- [Low-rate GPU investigation](results/GPU_INVESTIGATION.md).
- [Trading, queue modeling and arbitrage applicability](docs/TRADING_APPLICABILITY.md):
  a research direction, not a trading implementation or profit claim.
- [Current execution gates](docs/OWNERSHIP_PLAN.md) and broader
  [production roadmap](EXECUTION_PLAN.md).
- [Operations](OPERATIONS.md), [historical experiments](EXPERIMENTS.md), and
  [historical ARM cost board](COST.md).

This is a **local prototype**, not a production deployment. No real-user impact,
online engagement lift, public authentication, distributed failover or trading
profitability has been established. The final pilot ranker is a documented
heuristic. NVIDIA Compute Sanitizer remains blocked by the unapproved Windows
debugger setting; functional CUDA tests are not a substitute for that gate.

The raw TCP endpoint does not implement pilot eligibility. Use the pilot API for
the feedback contract, and keep both services private.
