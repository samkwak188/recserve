# Dependency diagnosis and recovery: 2026-10-02

Scope: local Windows/Ubuntu WSL and Docker, synthetic authenticated accounts,
real Caddy HTTPS, PostgreSQL, and C++ retrieval. No cloud deployment, external
alert delivery, HA, or online user impact is established.

## Changes and verification loops

1. Add private pool occupancy, bounded database failure reasons, retrieval
   latency/outcome histograms, and last reconciliation state. Hold all eight
   database pool slots and an account row lock: both fail with bounded 503s,
   then recover with rollback and exactly one preference effect after retry.
   Real socket tests cover malformed, disconnected, stalled and trickling peers.
   An explicit zero deadline now expires instead of receiving a fresh budget.
2. Force-kill retrieval, PostgreSQL and API containers while an independent
   HTTPS observer probes readiness. Verify eligible fallback, failure closure,
   restart, old-response replay, duplicate events and durable preferences.
   Preserve every probe, including failures, and upload samples on CI failure.
3. Diagnose and repair failures exposed by execution. Hosted Linux could not
   traverse the ledger fixture's mode-0700 temporary directory; individual
   certificate-file mounts preserve nonroot execution. A repeated local drill
   observed an 11.501-second ledger deletion failure, exceeding Caddy's ten-second
   write limit. Removing automatic SDK retries and tightening socket timeouts
   yielded an HTTP 503 in 3.795 seconds in the final drill. A real SDK retry-engine
   test verifies one attempt on a service 503.

## Final local evidence

- 39 PostgreSQL/API/C++ integration tests passed.
- Required pilot baseline: 28 Python tests, seven CTest suites, synthetic and
  existing MovieLens restart demos passed.
- Full container operations and encrypted backup/WAL restore stages passed.
  The restore recovered both the backed-up record and the later archived record;
  the wrong encryption key was rejected.

| Restarted component | Failed / attempted readiness probes | Restart to successful readiness |
|---|---:|---:|
| Retrieval | 6 / 8 | 0.482 s |
| PostgreSQL | 3 / 5 | 2.251 s |
| API | 2 / 4 | 3.213 s |

Timing begins immediately before the Docker start command after an observed
outage. The observer sends one HTTPS request at a time with a six-second timeout
and pauses 100 ms after each completion. Counts include the intentional outage.
These are sampled local recovery observations, not capacity, uptime or cloud
RTO estimates. Restarting the same containers preserves their addresses.
The separate fresh-volume encrypted restore took 3.744 seconds on two records;
model rollback took 0.187 seconds on this tiny synthetic fixture.

## Reproduce and inspect

Run the required Windows baseline with scripts/Check-Pilot.ps1. Run authenticated
checks with scripts/Run-Production.ps1 -Stage integration and then -Stage operations.
The production-pilot workflow runs both and the browser job on each push.

[Raw measurements](reliability-2026-10-02.json) contain code SHA-256 hashes,
original worktree source fingerprints, immutable image IDs, all final probe
samples, metric snapshots, command receipts, and failed-iteration references.
The measured worktree's parent is a0fa8c6; its source fingerprint and per-file
hashes identify the tested changes. Documentation added after measurement does
not silently relabel those receipts. Full local logs remain in .cache/production;
historical ARM, GPU and pilot result files are unchanged.

Database outage samples include client errors classified as "other"; the
bounded categories do not identify every libpq failure. Socket timeout settings
do not impose a universal DNS wall-clock bound. Cloud DNS, off-host restore,
alert delivery, long-duration load, live identity-provider configuration and
owner-led pilot use remain unqualified. CPU remains the default and the GPU
sanitizer gate remains unpassed.

## Hosted startup-race follow-up

At ec5f644, all 16 core jobs plus PostgreSQL/API and browser checks passed.
The hosted container job passed the new HTTPS recovery checks, then exposed an
older backup startup race: the default socket pg_isready probe accepted the
temporary initialization server before the application database existed.
All three test launchers and the Compose database health check now probe
127.0.0.1 explicitly, waiting for the final TCP listener.

The restore drill now deliberately pauses initialization. It asserts that the
old socket probe succeeds while the TCP probe fails, releases the pause, and
then completes encrypted backup, WAL recovery and wrong-key rejection.
This deterministic regression passed locally (runner exit 0), as did the full
operations stage with TCP readiness. [Separate follow-up evidence](readiness-regression.json)
preserves that run's source hashes. Earlier timing measurements above retain
their original source identity; the startup harness changes do not relabel them.
