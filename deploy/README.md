# Deployment boundary and operator runbook

This is a recoverable single-server pilot, not an HA service. The files here do
not provision anything. Do not expose a deployment until the remaining gates in
`docs/PRODUCTION.md` have passing, source-bound evidence and the owner approves
the actual initial and monthly charges (ceiling: $30/month).

## Local container verification

Run `scripts/Run-Production.ps1 -Stage operations`. This builds three images and
uses their immutable image IDs, not mutable tags. It tests real Caddy HTTPS,
the production API entrypoint, PostgreSQL migrations, the restricted runtime
role, two model/retrieval pairs, replacement/rollback, retries across replacement,
and database/retrieval/deletion-ledger failures. The S3-compatible HTTPS endpoint
is an isolated in-memory **test transport**, not off-host durability evidence.
Google sign-in is covered separately by controlled signed-token tests.

A separate pgBackRest drill encrypts a backup, archives a post-backup write,
kills PostgreSQL and restores into a fresh local volume. A wrong encryption key
must fail. Neither this tiny local dataset nor local Docker timing establishes
the cloud five-minute RPO or fifteen-minute restore-to-service target.

Full logs and JSON reports stay under `.cache/production`; only resources with
the run's unique ownership label are removed. Synthetic accounts, transient
credentials, and test databases are destroyed at the end of a drill. Failed
logs remain. No production data or existing legacy SQLite database is touched.

## Runtime configuration

`compose.production.yml` exposes only Caddy's HTTPS/HTTP ports. PostgreSQL,
retrieval and metrics have no published port. API, retrieval and web containers
run as UID 10001; PostgreSQL uses its image's postgres UID. Filesystems are
read-only, capabilities are dropped and memory/PID limits are explicit.

Supply immutable registry digests for `APP_IMAGE_BLUE`, `DATABASE_IMAGE`, and
`WEB_IMAGE`. The local builder produces image IDs for testing; CI must publish
approved release images before a real deployment. Do not build on the 2 GiB VM.
Supply `APP_HOSTNAME`, `GOOGLE_CLIENT_ID`, `MODEL_DIR_BLUE`, `ROUTING_DIR`,
`SECRETS_DIR`, `PRIVACY_S3_ENDPOINT`, `PRIVACY_S3_REGION`, and `PRIVACY_S3_BUCKET`.
The routing directory contains `upstream.caddy` with:

```caddyfile
reverse_proxy api-blue:8000
```

Mount secret files with these exact names, readable only by their intended
service UID and operator: `postgres_password`, `database_url`,
`migration_database_url`, `google_client_secret`, `privacy_access_key`,
`privacy_secret_key`, `privacy_fernet_key`, and `pgbackrest_config`. Keep the
directory outside the checkout and image build context. Never print rendered
Compose configuration, environment dumps, or a DSN into logs.

The owner migration URL and application URL must name the same database. The
application username must be `recserve_app`. Only the one-shot migration
container receives owner credentials. Rotate an existing runtime password with
an explicit coordinated operator action; migration does not silently rotate it.

Configure the Google callback as `https://<hostname>/auth/callback`. Require
verified invitations, a current consent version, and a genuine live sign-in by
the owner and one invited tester. No fixture identities are imported.

## Backup and deletion invariants

`pgbackrest.conf.example` is a template, not usable credentials. Use a scoped
off-host bucket and separately recoverable encryption key. Initialize the
stanza, take a full backup, verify WAL archival, then schedule and monitor daily
full backups. A seven-full-backup count is **not** seven-day retention when a
backup fails: configure and verify actual seven-day expiry with the provider.
Do not expire WAL needed by any retained backup. Alert on archive/backup age.

Use a separate restricted deletion ledger prefix/bucket and encryption key.
Retain tombstones longer than every restorable backup (pilot: at least eight
days for seven-day backups). Do not restore this ledger from the old primary
database snapshot. The API fails startup if replay cannot complete; do not
bypass it to make readiness green. Validate the bucket lifecycle and ability
to restore keys independently before collecting real personal data.

Restore on a fresh disposable host, replay deletions, verify authentication,
eligibility and a post-backup commit, and only then reopen the proxy. Record the
latest recovered commit timestamp and elapsed service restoration time. Delete
the disposable host only after preserving evidence and verifying its identity.

## Blue/green replacement and rollback

Keep the previous immutable application/model pair. Set `APP_IMAGE_GREEN` and
`MODEL_DIR_GREEN`, start the green profile privately, then verify its readiness,
model digest and authenticated smoke flow before routing traffic. Migrations
must remain compatible with both applications. Destructive cleanup is a later
release, never part of the switch.

Atomically replace `upstream.caddy` in the mounted routing directory with the
green upstream, validate Caddy configuration and reload Caddy. Verify fresh
requests use the new digest; old recommendation IDs must still replay the exact
original response. To roll back, restore the blue upstream and reload. Do not
roll back authoritative preferences or restore the database just to change a
model. Rehearse and record a complete rollback under five minutes.

Retrieval addresses are resolved at API startup, outside request deadlines.
Recreate/restart an API with its retrieval pair if the retrieval address changes;
do not replace a container address underneath a running pair.

The two-pair peak memory envelope is tight on 2 GiB. Measure host RSS and OOM
events during replacement, not only steady state; this local check does not
qualify that host. Do not silently upgrade the VM to make the target pass.

## Monitoring before invitations

Scrape `/internal/metrics` only on the private network. Public Caddy returns 404
for it. JSON API logs contain route templates, status, duration and random trace
IDs, never identities, cookies, emails or preference payloads. Configure host
journald retention at seven days and confirm rotation under disk pressure.
The application periodically expires 90-day raw interactions.

External HTTPS probing, alert delivery, WAL/backup age, disk/RAM, database pool
and retrieval queue monitoring still need a deployed collector and rehearsed
alert path. Missing monitoring is unknown, not uptime. Do not invite users
until these checks, cloud restore/rollback and the qualification campaign pass.
