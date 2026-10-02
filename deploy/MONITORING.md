# Monitoring and alert delivery

Run `scripts/Run-Production.ps1 -Stage monitoring`. This executes real Caddy,
PostgreSQL, API and retrieval containers, then Prometheus, Blackbox Exporter
and Alertmanager at the pinned image digests in monitoring/images.json.
The receiver is an isolated test fixture; no person or external service is
messaged. This is not a deployed external monitoring service.

## Executed contract

The collector must scrape the API through Caddy's private port 9000 while
public /internal/metrics remains blocked. This listener only routes the exact
metrics path; it uses the active blue/green upstream and the configured Host.
Never publish port 9000, 9090, 9093 or 9115 publicly.

The HTTPS probe requires certificate verification, HTTP 200 and a ready JSON
body. Redirects are failures. The drill stops Caddy, observes a firing
notification, restarts it and observes a resolved notification with the same
fingerprint and activation time. Assertions only accept notifications received
after the corresponding fault or restart. The test receiver rejects its first HTTPS outage notification with HTTP 503;
the same alert must subsequently be delivered successfully. Stopping the
prober must trigger a separate missing-observation alert and restarting it
must clear that alert. These checks exercise real evaluation and delivery.

The integration drill shortens scrape/evaluation periods to one second and
alert holds to three seconds. Nine promtool scenarios separately test the
unchanged production rules, including hold times, resolution, missing data,
sparse error traffic and failed notification delivery. Test notification times
are not promised deployment latencies. All attempts and intermediate assertions
remain in monitoring-attempts.json even when the drill fails.

## Rules and operator actions

| Signal | Hold | Response |
|---|---:|---|
| HTTPS not ready | 30 s | Check proxy, API readiness, database and retrieval. |
| Missing probe or private metrics | 1 min | Restore observation; missing data does not establish uptime. |
| Database pool at least 7/8 occupied | 5 min | Check locks and traffic before increasing pool limits. |
| Privacy reconciliation failed | 1 min | Repair ledger/database access; never bypass deletion replay. |
| At least five failed retrieval attempts in five minutes | 1 min | Inspect timeout/connection/protocol outcomes and degraded responses. |
| API 5xx above 5%, at least 20 requests in five minutes | 1 min | Diagnose dependencies and admission pressure. |
| Notification failures or missing Alertmanager | 1 min | Check the receiver through an independent channel. |

Thresholds are provisional diagnostic choices, not measured service SLOs.
Readiness probes count in retrieval telemetry; API error alerts exclude health
and metrics routes. A popularity-only workload may make no retrieval attempts.

## Deployment configuration boundary

`compose.monitoring.yml` joins an explicitly named existing private application
network. It uses nonroot containers, bounded memory/CPU, read-only roots, private
UIs and persistent bounded-retention state. Supply:

- RECSERVE_PRIVATE_NETWORK: the application's private Docker network.
- MONITOR_TARGETS_FILE: an absolute JSON file containing a Prometheus target
  group, for example `[{"targets":["https://your-host/readyz"]}]`.
- ALERT_WEBHOOK_FILE: an absolute private file containing the approved webhook
  URL, readable by UID 65534. Do not commit or log it. Use HTTPS for real delivery.

The target hostname must resolve from the prober. Public DNS/TLS are required
for the default probe; the test adds only its ephemeral CA and server name.
The collector adds up to 416 MiB of configured container memory limits. These
limits plus application/OS and blue/green peak memory need measurement on the
chosen host before accepting a 2 GiB deployment.

A same-host collector cannot reliably report loss of its own host. A separate
off-host probe and heartbeat/dead-man receiver are still required before
invitations. An alert about broken Alertmanager cannot reach the operator
through that same broken Alertmanager. No notification acknowledgement from a
human is claimed by the fixture's HTTP acknowledgement.

The cloud account, domain, approved notification destination and actual cost
remain owner configuration gates. Backup/WAL age, host disk/RAM pressure and
fresh-host restore require further deployment work. Do not interpret this
local pipeline as closing those gates.

## References

Configuration follows the official [alert rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/),
[rule tests](https://prometheus.io/docs/prometheus/latest/configuration/unit_testing_rules/),
[Alertmanager receiver configuration](https://prometheus.io/docs/alerting/latest/configuration/)
and [Blackbox Exporter configuration](https://github.com/prometheus/blackbox_exporter/blob/master/CONFIGURATION.md).
