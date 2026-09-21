# Local movie-discovery pilot

Purpose: return eligible movies, accept explicit feedback, and preserve its
effect across process restarts. This is a reference API for a future opt-in pilot,
not evidence that anyone finds its recommendations useful yet.

## Reproduction

PowerShell: `.\scripts\Check-Pilot.ps1`. Inside WSL: `bash scripts/check_pilot.sh`.
The file-based runner builds, runs unit/real-socket tests and a deterministic
demo. If `data/wsl_small_bundle.json` exists, it repeats the demo on trained
MovieLens embeddings. Full logs and timestamped receipts: `.cache/logs/pilot-*`.
The demo stops its own processes and removes only its temporary database/fixture.

To keep a local session running, start these in two WSL terminals from the repo:

```bash
python3 scripts/bundle.py serve --manifest data/wsl_small_bundle.json --binary build/recserve_server --backend hnsw
```

```bash
python3 -m recserve_pilot.service --manifest data/wsl_small_bundle.json --db .cache/pilot/feedback.sqlite --port 9410 --upstream-port 9400
```

Only `127.0.0.1` is supported. Use a non-browser local client with
`Content-Type: application/json`; browser Origins/unexpected Host headers are
rejected. These checks are **not authentication**. Do not expose this API or put
real personal data in it yet.

- `POST /v1/recommend`: `{"request_id":"r1","user_id":"1","k":5}`.
- `POST /v1/events`: the schema below; acknowledge `shown` before an outcome.
- `GET /healthz`: database/policy liveness, not retrieval-backend readiness.

```json
{
  "schema_version": 1,
  "event_id": "unique-event-id",
  "event_time_ms": 0,
  "user_id": "1",
  "item_id": "item-id-from-response",
  "kind": "shown",
  "request_id": "r1"
}
```

Replace the timestamp with current Unix epoch milliseconds. First send `shown`
for an item actually displayed, then `save`, `dismiss` or `watched` with another
event ID. Events may be up to 24 hours old or 30 seconds ahead and cannot
substantially predate their issued response. Exact committed retries remain
idempotent after aging out. Conflicting payloads or another ID for the same
impression/item/action return 409. Invalid inputs return 400; database contention
or repeated generation changes return 503, never a fabricated success.

MovieLens raw IDs select fixture users, not authenticated people. New demo users
must use `guest:` IDs and receive training-popularity cold start. A real identity/
consent flow must not assign people someone else's MovieLens history. Movie
titles/UI and real-user enrollment are not supplied.

## Decision record: policy ownership

The C++ server supplies candidate scores. `recserve_pilot` owns final eligibility,
the save-rate heuristic, and the durable journal. This closes a real network
response loop without adding database dependencies to every C++ benchmark.
It does **not** publish the C++ RCU feature table or integrate Kafka. Raw RSV1
still has no seen-item filtering; use the pilot API for this contract.

Warm users retrieve 64 candidates, doubling up to 512 when filtering leaves too
few. Remaining slots use labeled training-popularity fill. Upstream failure uses
the same eligibility policy with `degraded=true`; database failure is an error.
No eligible items means fewer than K with `exhausted=true`, never forbidden items.
ANN inclusion and the cap can change quality; this is not exact final top-K.

Final score: candidate score plus `0.05 * saved / (shown + 2)`. It is a transparent
heuristic, not learned or propensity-corrected CTR. Save/dismiss/watched also
exclude the item for that user. Issuing a response does not increment exposure;
`shown` does. Dismissal does not retrain ALS or update the user's embedding.

Online policy and replay callers share the eligibility function. This is policy
parity, not a new globally isolated offline evaluation. Popularity uses exported
training users only. All catalog items initially count as available;
`Store.set_unavailable` is a local operator hook, not a live inventory connector.

## Transaction and recovery contract

SQLite WAL + FULL synchronization commits event ID/payload, sequence, projection
and generation together. A retry finds the original commit or reapplies the
uncommitted operation. Process-death tests use `os._exit` before/after commit;
independent journal replay checks the projection. A test restores a SQLite online
backup to another file. These tests do not prove power-failure, disk-loss or
cross-host durability. WSL tests use the workspace's mounted Windows filesystem;
production storage/backup semantics require validation on the actual target.

Retrieval occurs outside a transaction. Response publication checks the captured
generation inside a write transaction and retries up to three times if feedback
won the race. A **new** request after feedback acknowledgement sees that state.
A reused request ID deliberately returns its original response: use a new ID for
fresh recommendations. Publication does not prove delivery/display.

The database is bound to the manifest hash. Switching models fails closed; use
an explicit migration or a separate database, never a silent reset. Model files
must remain immutable while running; hashes are not signatures.

## Remaining release gates

- Authentication, per-user authorization, consent, retention/deletion, TLS and
  an actual opt-in workflow. Local IDs are not access controls.
- Supported HTTP deployment stack. This standard-library local server bounds
  active handlers at eight with a 16-connection backlog; it is not internet-ready.
- Whole-pilot load/resource tests, database maintenance, disk-full/machine-loss
  drills, storage retention limits, and model migration.
- Exposure-quality audit, global train/validation/test isolation, cohort results/
  confidence intervals, and a learned ranker if real data justifies it.
- A chosen outcome and opt-in user feedback. Acceptance tests do not establish
  engagement lift or readiness for real users.
