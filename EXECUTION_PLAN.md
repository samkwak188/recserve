# RecServe execution plan

Baseline: 86bbbc22462c70ff5fa2307dadb161183bf9a324, 2026-09-20.
Target: a reproducible recommendation service with measured quality, durable
features, deadline-aware CPU/GPU retrieval, and an operational deployment.
The existing Ubuntu 24.04 WSL environment exposes the RTX 3060 (12 GiB, sm_86).
Use Linux GCC/CMake/CUDA for the primary build; retain CPU-only portability.

## Execution and evidence rules

- Keep commands in scripts; keep full outputs in `.cache/logs/` and report exit
  status plus concise summaries. Benchmark artifacts must include source commit,
  platform, compiler, actual ISA, fixture hashes, parameters, and timestamp.
- Separate historical ARM results from new x86/WSL results. Never relabel merged
  measurements from another host. Load deterministic fixtures during comparisons.
- Gate each phase on executable checks. Update status with evidence as it lands.
- Infrastructure spending, public deployment, and third-party account setup need
  actual credentials/budget; prepare and verify local artifacts first.

## P0: reproducible foundation (implemented and verified)

1. Bootstrap existing Ubuntu WSL with CMake, Ninja, Python venv, librdkafka, and
   CUDA 13.1 compiler/runtime/cuBLAS development packages. Keep Windows driver.
2. Build Release CPU-only and run the existing suite before modifying behavior.
3. Add scripted checks, dependency manifests, and separate host-specific results.
4. Repair protocol bounds, configuration validation, artifact compatibility,
   snapshot lifetime safety, and misleading benchmark controls.
5. Fix CI regression comparisons to measure base and candidate on one runner;
   test the gate with a deliberately bad result and a missing baseline.

Acceptance: CPU tests pass; malformed inputs are rejected; historical evidence is
preserved; reported backend/ISA matches executed code; negative gates fail.

## P1: GPU retrieval (runtime verified; sanitizer gate pending)

1. Optional CUDA build with a tested unavailable stub and explicit error behavior.
2. Persistent catalog upload and bounded, reusable workspace; FP32 cuBLAS batched
   inner products; tile the catalog; select top-k on device; transfer only top-k.
3. Test odd dimensions, tail tiles, batch sizes, ties, invalid input, concurrent
   callers, repeated calls, and agreement against CPU exact retrieval.
4. Add batch-aware benchmark and CPU exact/int8/HNSW comparisons. Record device
   time and end-to-end latency separately; include warmup and repeated processes.
5. Sweep items 4K/16K/64K/256K/1M and batches 1/8/32/128/512; use 2048 only within
   workspace limits. Publish measured crossover, recall, throughput, and VRAM.
6. Add CUDA compile CI and explicit GPU execution workflow; protect any owner
   runner from untrusted pull requests. CPU tests must work without CUDA.

Acceptance: exact IDs for separated scores; documented FP32 tolerance; bounded
allocation; actual RTX execution and sanitizer evidence; no invented cloud cost.

## P2: usable service (local runtime implemented; pilot gates remain)

1. Load compatible catalog/index/query/model bundles with stable external IDs.
2. Bounded connections and work queues, validated framing, read/write deadlines,
   graceful drain, configurable bind, health/readiness and metrics.
3. Deadline-aware GPU batching with CPU fallback and explicit overload behavior.
4. True open-loop load generator with bounded concurrency, attempted/completed/
   failed accounting, latency from intended arrival, and request deadlines.
5. Network smoke, malformed-client, stalled-client, overload and shutdown tests;
   one-hour soak and sustained arrival-rate sweep.

Acceptance: establish measured capacity at a provisional 5 ms p99 target, record
quality and failure rates at each point, and recover from overload. 1,000 QPS over
1M items is an experiment target, not a promised result.

## P3: durable data and recommendation value

1. Version events with IDs, timestamps, partition offsets and schema; define
   view/click attribution, deduplication and invalid-record handling.
2. Checkpoint feature state and offset vector atomically; restore and replay;
   test crash before/after checkpoint, duplicates, rebalances and multiple partitions.
3. Implement equivalent event-time windows online and offline, late-data policy,
   cold-item smoothing, feature TTL, and explicit freshness/skew SLOs.
4. Connect the serving process to the feature stream; validate event-to-response.
5. Preserve external IDs in training exports; use temporal train/validation/test
   splits; train and evaluate a ranker with popularity and ALS-only baselines.
6. Add seen/availability filters, cold-start fallback, diversity and impression
   logging with model/feature versions and experiment assignment.

Acceptance: deterministic recovery without lost/doubled counts, window equality
on adversarial event fixtures, measured freshness, and held-out quality comparison
with confidence intervals. Offline lift does not establish online user impact.

## P4: deployable pilot and distributed validation

1. CPU/GPU container images, one-command local stack, pinned dependencies,
   telemetry dashboard, documented SLOs, rollback and incident runbooks.
2. Canary artifact promotion, health-based routing, backup/restore and failure drills.
3. Networked shards/replicas with timeout budgets, partial-result policy and
   failover; validate kill/restart during traffic before making scale claims.
4. Measure the actual target cloud instance before modeling dollars; include
   effective throughput at SLO, replica topology, RAM/VRAM, storage and network.
5. Pilot with real opt-in users; evaluate completion/engagement and guardrails.

Acceptance: reproducible deployment and rollback, recovery evidence, load/SLO
report, and an honest case study suitable for a resume with measured claims only.

## Executed checkpoint: 7e0eb63

Telemetry/dependency follow-up: `4651d0d`, also 16/16 hosted jobs passed
(run 35542148547). It separates normal connection EOF from I/O failures;
the original capacity artifact retains its measured counters with that caveat.

- Bootstrapped Ubuntu 24.04 WSL, GCC 13.3, CMake/Ninja, CUDA 13.1, cuBLAS,
  Compute Sanitizer, librdkafka, and a repository-local Python environment.
- Baseline CPU tests passed before changes. Current CPU-only and AVX2 builds,
  address/undefined/thread sanitizer suites, malformed artifact tests and real
  socket tests pass. WSL TSan requires disabling ASLR for the test process tree;
  no system-wide ASLR policy was changed.
- CUDA correctness, concurrency, tile tails, ties and real socket batching pass
  on the RTX 3060. Hosted CUDA compilation passes; the manually dispatched GPU
  runner workflow is configured but no persistent owner runner was registered.
- Added bounded connection admission, frame deadlines, graceful stop, private
  health/metrics, deadline-aware GPU microbatching and exact CPU circuit-breaker
  fallback. Injected device-failure tests verify fallback without changing recall
  semantics. Existing rankd/shard executables remain experimental, not deployment
  entry points.
- Repaired RSS, misleading pin/arena controls, malformed framing, reader slot
  ownership, artifact bounds and the reference-validation query mismatch.
  Performance CI measures base and candidate on the same runner. Negative gates
  reject missing measurements, regressions and invalid numbers.
- Re-trained MovieLens-small, preserved raw identity maps, verified C++/NumPy
  agreement, and created integrity-checked serving bundles. This is still the
  original ALS/per-user-holdout evaluation, not a newly trained production ranker.
- Built and tested a non-root CPU image; a real-model TCP request and health check
  passed under read-only filesystem, PID limits and loopback-only port publishing.
  The temporary validation container/network were removed; model files remain.
- All 16 hosted jobs passed for the runtime checkpoint:
  https://github.com/samkwak188/recserve/actions/runs/35541070732
- Measurements and remaining limitations are recorded in `results/` and
  `OPERATIONS.md`. Historical ARM artifacts are preserved.

The following tickets are ordered release gates, not a claim that P3/P4 are done.

## Remaining work: precise implementation tickets

### R1. Close the GPU release gate

Dependency: Windows administrator approval for NVIDIA's debugger interface.

1. Enable the documented GPU debugger interface only with owner approval; rerun
   `scripts/check_gpu.sh`. Require Compute Sanitizer exit 0 and zero reported
   memory errors. A successful functional oracle is not a sanitizer substitute.
2. Add initcheck/synccheck to trusted GPU validation after memcheck works. Record
   driver/toolkit/device and binary hash with every run.
3. Register an ephemeral or manually started GPU runner only with approval. Never
   execute untrusted PR code on the workstation. Keep hosted compile validation
   separate from actual GPU execution.
4. Extend the sweep to the chosen production embeddings and candidate count.
   The current crossover is top-10 retrieval; serving retrieves 64 or more before
   ranking. Require matched quality, memory headroom, repeated processes and
   arrival-rate latency, not preformed-batch QPS alone.
   Add an all-core CPU and CPU batched-GEMM baseline; the present comparison is
   explicitly single-threaded AVX2 SIMD, not the best possible whole-host CPU.
5. Use a measured routing policy: small/light traffic stays CPU; GPU batching is
   enabled only where measured useful at the service SLO. Do not assume desktop
   RTX measurements transfer to a cloud GPU or quote an invented dollar saving.

Done when: memory/race tools pass on hardware, real-data retrieval agrees within
declared tie tolerance, and a reproducible latency/quality/cost decision exists.

### R2. Establish the service contract and capacity envelope

Dependencies: R1 for GPU promotion; pilot choice for business-facing semantics.

1. Run a one-hour sustained soak after the short capacity sweep, then overload
   at 2x/4x measured capacity and recover. Record RSS/VRAM trend, accepted/rejected
   connections, completed requests, deadline failures and thread/FD counts.
2. Test GPU failure and fallback over sockets, slow readers, byte-drip clients,
   disconnects during writes, shutdown during a full queue and repeated restarts.
   Unit fault injection currently covers fallback; it is not a hardware-failure
   drill. CPU/GPU work is non-preemptible: expiration suppresses results, not
   arbitrary running instructions.
3. Add server histogram buckets for queue, batch formation, retrieval, ranking
   and total duration; label backend/model version with bounded cardinality.
   Existing counters and client latency artifacts are useful but not a complete
   operational telemetry stack.
4. Set an initial availability/error budget from observed pilot traffic. The
   provisional laboratory target is p99 <=5 ms and <=0.1% failures. Count client
   queue drops and transport failures, not only successful-request percentiles.
5. Introduce a versioned external API mapping raw user/item IDs to bundle rows;
   reject unknown identities or use an explicit cold-start policy. Current TCP
   IDs are dense internal rows. Never apply modulo identity mapping in a public
   API. The server now rejects out-of-range rows.

Done when: a capacity curve includes quality and all failed arrivals, resource use
is stable, overload recovers, and client-visible semantics are documented/tested.

### R3. Build one durable event-to-response vertical slice

Dependency: choose movie/media, commerce, or learning/content pilot. The existing
MovieLens path is a test fixture, not evidence of a real product need.

1. Define schema v1: `event_id`, `event_time_ms`, `ingest_time_ms`, stable user/item
   IDs, `event_type`, `request_id`, `impression_id`, rank, model/experiment version.
   Add JSON/schema validation, bounded payloads and a quarantined invalid stream.
   Choose the actual success event with the pilot; do not reinterpret MovieLens
   positive ratings as logged impressions or unbiased CTR labels.
2. Disable consumer auto-commit. Persist aggregate changes, event deduplication
   keys and the partition-offset vector in one transaction. A local SQLite/WAL
   reference is sufficient for crash tests; a shared production store needs
   ownership/fencing and operational backup policy. Broker commits are not the
   sole recovery source of truth.
3. On restart restore state and seek each assigned partition to the committed
   offset plus one. On rebalance fence the previous owner before accepting new
   writes. Test crash before commit, after commit/before broker acknowledgement,
   duplicate delivery, offset gaps, two partitions, reassignment and replay.
4. Define event-time windows and late/future-data policy once. Suggested pilot
   windows: 5 min/1 h/24 h; 5 min allowed lateness; future-skew records quarantined.
   Make these configuration, not hard-coded business assumptions. Test boundary
   timestamps, out-of-order records, duplicate IDs, expiry and empty windows
   against an independently implemented offline reference.
5. Publish a versioned snapshot atomically only after the state transaction.
   Include offset vector, event-time watermark and feature schema version. Wire
   this snapshot into `recserve_server`; currently the nearline demonstration
   and deployable server are separate processes with no durable bridge.
6. Verify: accepted event -> committed offset -> published version -> changed
   recommendation score on a known query. Add stale-feature readiness policy,
   bounded fallback and freshness/skew alerts. Define and measure the pilot's
   freshness target; do not fabricate a latency guarantee before this path exists.

Done when: a recorded event changes a served response, restart/replay produces
identical features without double counting, and stale inputs degrade explicitly.

### R4. Make ranking useful, not merely fast

Dependencies: R3 schema, pilot objective and legally usable data.

1. Add train/validation/test time boundaries with global time isolation, plus
   warm-user, cold-user and cold-item cohorts. Keep the current per-user split as
   a separately labeled legacy benchmark. Fit encoders/popularity/normalizers on
   training only. Store dataset hash, seed, package versions and cutoff times.
2. Train a small interpretable ranker on logged exposure/outcome features. Compare
   popularity, ALS-only, fresh-feature reranking and the trained ranker under the
   same candidates/filters. Current weights are hand-set; the user-CTR term is
   constant across a user's candidates and cannot improve their ordering by itself.
3. Apply seen-item, availability and policy filters in serving, with sufficient
   candidate replenishment to return K eligible items. The evaluator filters seen
   items after scoring; the current server does not yet have that parity. Add
   explicit popularity/content cold start and diversity guardrails.
4. Report Recall/NDCG plus catalog coverage, novelty/diversity and latency. Use
   paired per-user bootstrap confidence intervals; select hyperparameters on
   validation only. A candidate is promoted only if its agreed metric improves
   without unacceptable cohort/latency regressions. No mandated fake lift.
5. Log impressions and model/feature versions before introducing experiments.
   Obtain opt-in pilot users, document deletion/retention, and randomize a small
   controlled rollout. Only this stage can support a real user-impact claim.

Done when: offline/online feature and filter parity holds, quality comparisons are
reproducible, and actual pilot outcomes—not synthetic throughput—justify adoption.

### R5. Promote a recoverable pilot

Dependencies: R2-R4 and explicit target environment, credentials and budget.

1. Use immutable image digests and verified artifact bundles. Bundle hashes detect
   changes but are not signatures or a semantic quality gate; authenticate the
   promotion channel and run quality checks before marking a bundle ready.
2. Add GPU runtime image validation, a real local Kafka/feature/serving composition,
   private metrics scraping/dashboard, authenticated TLS ingress, rate limits,
   resource limits and vulnerability/license checks. The legacy Flink compose
   configuration is not the production stack.
3. Launch old/new bundles side by side, shadow requests, compare eligibility and
   quality, then canary. Drill rollback by switching the immutable bundle/image,
   not editing a live artifact. Drill restoration from an actual feature backup.
4. Add networked shards only after a single host's measured limits require them.
   Define timeout budgets, partial-result policy, replica ownership and failover;
   kill/restart a shard during traffic. Existing in-process scatter/gather numbers
   do not prove distributed fault tolerance.
5. Benchmark the exact chosen cloud instance. Model effective throughput at SLO,
   two-replica availability, model/feature storage, network and operational cost.
   Do not multiply desktop QPS by an unrelated instance price.

Done when: deploy/rollback/restore can be reproduced by another engineer, SLOs and
data retention are operationally enforceable, and a pilot outcome report exists.

## Resume/case-study evidence policy

- Already defensible: implemented/tested CUDA exact retrieval and deadline-aware
  batching; measured CPU/GPU crossover; repaired a false-green performance gate;
  verified Windows/Linux builds and sanitizer coverage; served a hashed real-data
  model from a bounded, non-root container.
- Qualify every number by hardware, batch, data size, retrieval quality and whether
  it includes queue/network latency. Keep benchmark QPS distinct from service QPS.
- Not yet defensible: production deployment, end-to-end exactly-once features,
  online engagement lift, cloud cost savings, distributed HA or million-item
  1,000-QPS service at the proposed SLO.
