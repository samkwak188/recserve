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
