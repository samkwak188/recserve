# Coherent pilot execution

Scope: close one movie-feedback loop before expanding infrastructure. Keep all
execution in versioned commands, full logs in `.cache/logs`, and concise evidence
in `results`. Existing ARM and RTX benchmark results are not overwritten.

## Ordered acceptance gates

1. **Durable pilot (in progress).** A loopback API owns external identities,
   impressions, feedback, eligibility and final policy. The existing C++ process
   retrieves candidates over RSV1. SQLite transactions couple deduplication,
   feedback projection and generation. This is a single-host reference, not the
   Kafka consumer or a distributed exactly-once guarantee.
2. **Recovery and parity.** Test actual process death before/after commit,
   duplicate/conflicting events, concurrent requests, model mismatch, stale
   request snapshots, cold start, training-seen exclusion, replenishment, upstream
   failure and restart against the real TCP server. Share one final eligibility
   function between replay evaluation and online use.
3. **Reproducibility.** Add a Windows entry script and platform-independent demo;
   integrate tests with CTest/hosted CI. Capture a real MovieLens demo separately
   from synthetic correctness fixtures. No dataset download is needed for CI.
4. **GPU investigation.** Add bounded stage measurements; repeat cold/warm and
   batch-policy experiments at low traffic. Do not infer the root cause from a
   single sweep. Keep CPU default until evidence supports a different policy.
5. **Trading applicability.** Produce a source-backed component mapping and a
   deterministic offline queue/arbitrage-assumption test harness if useful.
   No broker connections, orders, financial-data purchases or profit claims.
6. **Public scope.** Make the README describe supported core versus experiments;
   document remaining gates and reproducible commands, not a feature checklist.

## Explicitly not finishable by code alone

Recruiting opt-in users, selecting outcome/retention policy, user comprehension
and review, financial data rights, live deployment budget, GPU debugger approval
and actual online impact require owner decisions or external evidence. A local
acceptance run does not mark these complete.

## Deferred until the loop is validated

New sharding, Kubernetes, cloud price claims, automatic GPU routing, a learned
ranker without exposure labels, and expanding the core into an order-entry
system. Multicore/GEMM quality-matched benchmarking and global temporal model
evaluation remain separate experiments, not consequences of passing the pilot.
