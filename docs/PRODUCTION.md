# Authenticated movie pilot: implementation and release ledger

This successor to the local SQLite demonstration is under construction. It is
not deployed and has no real-user, availability, or cloud-capacity evidence.
Historical measurements in `results/` retain their original scope.

## Execution

Windows: `scripts/Run-Production.ps1 -Stage fast`. The script uses Ubuntu WSL and
the repository-local `.cache/venv-production` environment. Commands live in
`scripts/production_stages.json`; full logs and source-bound JSON receipts live
in `.cache/production/runs`. `-Resume` requires matching source, inputs, tool and
dependency versions and untampered successful logs. Missing stage implementations
fail, never skip green. No secrets belong in command arguments or test output.

## Ordered gates

0. Source-bound execution and existing baseline checks.
1. PostgreSQL accounts, Google OIDC, ownership, consent, privacy and migrations.
2. Item-only bundles, vector retrieval, preference fold-in and quality evaluation.
3. Browser application and generated API types.
4. Container deployment, monitoring, off-host backup and recovery.
5. Source-bound security, quality and full-stack load qualification.
6. Approved cloud deployment, real OAuth and fresh-host restore/rollback.
7. Owner-led invitations and 30 days of observed pilot use.

Each implementation checkpoint records what ran and what remains. No cloud
resources, invitations, trading connections or GPU debugger policy changes are
authorized by a local test. The monthly budget ceiling is $30; account/domain,
OAuth, backup, alert destination and actual billing approval gate deployment.

## Product boundary

Invite-only, consenting adults; noncommercial historical movie discovery.
MovieLens-small snapshots retain the GroupLens research license and attribution.
MovieLens 25M stays a separate local evaluation dataset unless permission is
resolved. No posters, streaming availability, payments, ads or fabricated lift.
The selected release policy is the best validated baseline, including popularity.

## Accounts checkpoint

`bash scripts/setup_production.sh` installs the complete hash-locked Python
environment. `-Stage integration` starts an isolated PostgreSQL container,
applies Alembic migrations and runs real-database API and signed-token OIDC
checks. It removes only that run's container, including its temporary test data.

Google credentials and a PostgreSQL URL are mandatory startup configuration;
there is no development login endpoint or anonymous identity fallback. Tests
use ephemeral signed tokens and controlled provider responses, not real Google
credentials. Production cookies require HTTPS even in local browser workflows.

Account deletion immediately removes primary records and sessions and commits a
deletion outbox entry. Off-host delivery and restore fencing are not yet wired;
therefore this checkpoint is not cleared for real personal data. Export is a
synchronous authenticated snapshot. Raw event retention requires scheduling the
maintenance task before launch. Live Google configuration remains an owner gate.

## Item-only personalization checkpoint

The v2 API accepts explicit likes/dislikes, maintains per-account revisions,
retrieves with an independently verified fold-in vector, filters all explicit
preferences and saved/watched/dismissed movies, and records attributed feedback
transactionally. Identical recommendation request IDs replay for 24 hours;
expired IDs return 409 rather than being silently repurposed. Fresh requests
recheck the account revision before committing an impression. Unrelated users
do not invalidate each other's snapshots. The popularity fallback requires
authoritative database state and labels retrieval failures as degraded.

`scripts/model_v2.py` verifies or launches an item-only bundle. Its model digest
is the manifest SHA-256, which covers all artifact hashes. Launch only through
this verifier with immutable files; the C++ process checks the supplied identity,
not the cryptographic provenance of a manually supplied command-line digest.
RSV1 and historical bundles remain unchanged. RSV2 is CPU-only in this release;
attempting vector-only CUDA serving fails explicitly.

RSV2 wire values are little-endian. Frame: uint32 magic `0x52535632`, uint32
payload length. Request payload: uint64 request ID, uint32 remaining timeout in
microseconds (1..1,000,000), uint32 dimension (1..4096), uint32 count (1..512),
32-byte model digest, then dimension float32 values. Response payload: uint64
request ID, uint32 status, uint32 count, 32-byte actual model digest, then count
pairs of uint32 dense item row and float32 score. Only the authenticated API
maps these rows to movie IDs. Never publish the raw retrieval port.

The PostgreSQL/C++ test suite covers numerical fold-in parity, ANN candidate
recall on synthetic vectors, malformed frames, model mismatch, account-scoped
idempotency, concurrent writes, revision races and database-failure closure.
This does not establish MovieLens quality, full-stack capacity or deployment.

## Chronological quality evaluation

Install `requirements-training.lock` into the local execution environment with
`pip install --require-hashes -r requirements-training.lock`, then run the
`quality` stage. Training has global 80/90-percentile timestamp cutoffs; ties
stay on one side. Every fifth raw user ID is withheld from model fitting.
Held-out users supply five earlier positive onboarding events; later events
alone become targets. Training-only encoders, popularity and item factors are
used by all policies. Unsupported targets and omitted users are counted.

Compare popularity, liked-item centroid and ALS on validation; test only the
selected candidate against popularity. A positive lower bound of a 1,000-draw
paired bootstrap NDCG@10 interval is required for personalized promotion.
The candidate retrieval/filtering path is shared with serving and uses the real
RSV2 process. ANN recall@128 must be at least 0.98. Results include cohort sizes,
Recall/NDCG, coverage and popularity concentration. These are observational
development-data measurements, not causal impact or research-benchmark claims.

The first small-snapshot run selected popularity: ALS's held-out NDCG difference
interval crossed zero. Raw reports and immutable bundles are under
`.cache/production/quality`; a compact source-bound summary is published in
`results/production-quality-small.json`. No old benchmark is overwritten.

## Browser application

The React/Vite application is under `web/`. OpenAPI is generated with
`scripts/export_openapi.py`; `npm run types` generates TypeScript interfaces.
`npm ci`, `npm run build` and `npx playwright install chromium` prepare the
browser stage. Windows uses `scripts/Check-Web.ps1` (a process-local execution
policy override is used by the runner; no machine-wide policy is changed).

The `browser` stage runs Chromium against real HTTPS, PostgreSQL and C++ with
ephemeral synthetic accounts. It covers consent, five explicit preferences,
recommendations, visible-impression attribution, saving, retry idempotency,
watchlist, export, deletion, mobile empty states, logout and keyboard access.
Generated screenshots are under `.cache/production/browser/<run>/`; they are
test-fixture views, not real users or a public deployment. The fixture's
self-signed certificate exception is limited to Playwright test configuration.
Production uses Caddy and trusted HTTPS certificates.
