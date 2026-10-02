"""Real monitoring pipeline exercised only against owned local test containers."""
import re
import hashlib
import json
import os
from pathlib import Path
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def verify_monitoring(run, start, bind, fixture, certificate, directory, web):
    config_dir = ROOT / 'deploy/monitoring'
    config = fixture / 'monitoring'
    config.mkdir()
    # Adjust only test clocks and the fixture's trusted CA. Native tools validate YAML.
    rules = re.sub(r'(?m)^(\s+for:) [^\n]+', r'\1 3s', (config_dir / 'rules.yml').read_text())
    blackbox = (config_dir / 'blackbox.yml').read_text() + (
        '      headers:\n        Host: localhost\n'
        '      tls_config:\n        ca_file: /etc/blackbox/ca.crt\n        server_name: localhost\n')
    prometheus = (config_dir / 'prometheus.yml').read_text()
    for before, after in [('scrape_interval: 15s', 'scrape_interval: 1s'),
                          ('evaluation_interval: 15s', 'evaluation_interval: 1s'),
                          ('scrape_timeout: 6s', 'scrape_timeout: 800ms')]:
        assert before in prometheus
        prometheus = prometheus.replace(before, after)
    am = (config_dir / 'alertmanager.yml').read_text().replace('group_wait: 10s', 'group_wait: 1s')
    am = am.replace('group_interval: 30s', 'group_interval: 1s')
    for name, value in [('rules.yml', rules), ('blackbox.yml', blackbox),
                        ('prometheus.yml', prometheus), ('alertmanager.yml', am)]:
        (config / name).write_text(value)
        (config / name).chmod(0o644)
    (config / 'https-targets.json').write_text(json.dumps([{'targets': ['https://web:8443/readyz']}]))
    (config / 'https-targets.json').chmod(0o644)
    webhook = config / 'alert_webhook_url'
    webhook.write_text('http://monitor-receiver:8088/alerts')
    webhook.chmod(0o644)

    run(['docker', 'compose', '-f', 'compose.monitoring.yml', 'config', '--quiet'],
        env=dict(os.environ, MONITOR_TARGETS_FILE=str(config / 'https-targets.json'),
                 ALERT_WEBHOOK_FILE=str(webhook), RECSERVE_PRIVATE_NETWORK='validation-only'))
    locks = json.loads((config_dir / 'images.json').read_text())
    mounts = [*bind(config_dir, '/rules')]
    run(['docker', 'run', '--rm', '--network', 'none', '--read-only', '--tmpfs', '/tmp:size=128m', '--cap-drop', 'ALL',
         '--entrypoint', '/bin/promtool', *mounts, locks['prometheus']['digest'],
         'test', 'rules', '/rules/rules.test.yml'])
    # amtool validates the actual receiver configuration, including the mounted URL file.
    run(['docker', 'run', '--rm', '--network', 'none', '--read-only', '--tmpfs', '/tmp:size=128m', '--cap-drop', 'ALL',
         *bind(config / 'alertmanager.yml', '/etc/alertmanager/alertmanager.yml'),
         *bind(webhook, '/run/secrets/alert_webhook_url'),
         '--entrypoint', '/bin/amtool', locks['alertmanager']['digest'],
         'check-config', '/etc/alertmanager/alertmanager.yml'])
    receiver = start('monitor-receiver', 'app', ['--entrypoint', 'python',
        '-p', '127.0.0.1::8088', *bind(ROOT / 'tests/production/alert_receiver.py', '/receiver.py')],
        ['/receiver.py'])
    start('alertmanager', 'alertmanager', [
        *bind(config / 'alertmanager.yml', '/etc/alertmanager/alertmanager.yml'),
        *bind(webhook, '/run/secrets/alert_webhook_url'), '--tmpfs', '/alertmanager:uid=65534,gid=65534,size=16m'],
        ['--config.file=/etc/alertmanager/alertmanager.yml', '--storage.path=/alertmanager',
         '--cluster.listen-address='])
    blackbox_container = start('blackbox', 'blackbox', [
        *bind(config / 'blackbox.yml', '/etc/blackbox/blackbox.yml'),
        *bind(certificate, '/etc/blackbox/ca.crt')], ['--config.file=/etc/blackbox/blackbox.yml'])
    prom = start('prometheus', 'prometheus', ['-p', '127.0.0.1::9090',
        *bind(config / 'prometheus.yml', '/etc/prometheus/prometheus.yml'),
        *bind(config / 'rules.yml', '/etc/prometheus/rules.yml'),
        *bind(config / 'https-targets.json', '/etc/prometheus/https-targets.json'),
        '--tmpfs', '/prometheus:uid=65534,gid=65534,size=64m'],
        ['--config.file=/etc/prometheus/prometheus.yml', '--storage.tsdb.path=/prometheus',
         '--storage.tsdb.retention.time=1h'])

    def origin(container, port):
        inspection = json.loads(run(['docker', 'inspect', container], True))[0]
        published = inspection['NetworkSettings']['Ports'][str(port) + '/tcp'][0]['HostPort']
        return 'http://127.0.0.1:' + published

    prom_url, receiver_url = origin(prom, 9090), origin(receiver, 8088)
    observations = []
    with httpx.Client(timeout=3, trust_env=False) as client:
        def query(expression):
            response = client.get(prom_url + '/api/v1/query', params={'query': expression})
            response.raise_for_status()
            data = response.json()
            assert data['status'] == 'success'
            return data['data']['result']

        def attempts():
            response = client.get(receiver_url + '/attempts')
            response.raise_for_status()
            return response.json()

        def delivered(name, state, since_ms):
            return [dict(fingerprint=a['fingerprint'], starts_at=a['starts_at'], received_ms=item['received_ms'])
                    for item in attempts() if item['accepted'] and item['received_ms'] >= since_ms
                    for a in item['alerts'] if a['labels']['alertname'] == name and a['status'] == state]

        def wait_for(label, predicate, timeout=60):
            started = time.monotonic()
            while time.monotonic() - started < timeout:
                try:
                    value = predicate()
                    if value:
                        observations.append(dict(check=label, elapsed_s=time.monotonic() - started))
                        return value
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(.2)
            raise RuntimeError('Monitoring assertion timed out: ' + label)

        try:
            wait_for('healthy HTTPS and private API scrape', lambda:
                query('probe_success{job="recserve-https"} == 1')
                and query('up{job="recserve-api"} == 1'))
            assert query('recserve_database_pool_capacity{job="recserve-api"} == 8')
            wait_for('initial probe alerts cleared', lambda: not query(
                'ALERTS{alertname=~"RecServeProbeMissing|RecServeHTTPSUnavailable"}'))
            outage_ms = time.time_ns() // 1000000
            run(['docker', 'kill', web])
            firing = wait_for('HTTPS failure delivered', lambda: delivered('RecServeHTTPSUnavailable', 'firing', outage_ms))
            def retried():
                data = [item for item in attempts() if item['received_ms'] >= outage_ms]
                failed = {a['fingerprint'] for item in data if not item['accepted'] for a in item['alerts']}
                succeeded = {a['fingerprint'] for item in data if item['accepted'] for a in item['alerts']}
                return sorted(failed & succeeded)
            retried_fingerprints = wait_for('rejected webhook retried', retried)
            restart_ms = time.time_ns() // 1000000
            run(['docker', 'start', web])
            resolved = wait_for('HTTPS recovery delivered', lambda: delivered('RecServeHTTPSUnavailable', 'resolved', restart_ms))
            assert {(a['fingerprint'], a['starts_at']) for a in firing} & {(a['fingerprint'], a['starts_at']) for a in resolved}
            probe_outage_ms = time.time_ns() // 1000000
            run(['docker', 'kill', blackbox_container])
            missing = wait_for('missing probe delivered', lambda: delivered('RecServeProbeMissing', 'firing', probe_outage_ms))
            probe_restart_ms = time.time_ns() // 1000000
            run(['docker', 'start', blackbox_container])
            cleared = wait_for('probe recovery delivered', lambda: delivered('RecServeProbeMissing', 'resolved', probe_restart_ms))
            assert {(a['fingerprint'], a['starts_at']) for a in missing} & {(a['fingerprint'], a['starts_at']) for a in cleared}
            report = dict(scope='Local containers and isolated webhook fixture; no external operator notified.',
                timings='Test-only 1s scrape/evaluation and 3s alert holds. Production holds are unit tested unchanged.',
                tls_verified=True, private_metrics_scraped=True, receiver_retry_verified=True,
                outage_to_delivery_s=(firing[0]['received_ms'] - outage_ms) / 1000,
                restart_to_resolved_s=(resolved[0]['received_ms'] - restart_ms) / 1000,
                matched_retry_fingerprints=retried_fingerprints,
                probe_outage_to_delivery_s=(missing[0]['received_ms'] - probe_outage_ms) / 1000,
                probe_restart_to_resolved_s=(cleared[0]['received_ms'] - probe_restart_ms) / 1000,
                missing_probe_detected=True, production_rule_tests=9, observations=observations,
                production_rules_sha256=hashlib.sha256((config_dir / 'rules.yml').read_bytes()).hexdigest(),
                test_rules_sha256=hashlib.sha256((config / 'rules.yml').read_bytes()).hexdigest(),
                images=locks)
            (directory / 'monitoring-report.json').write_text(json.dumps(report, indent=2) + '\n')
            return report
        finally:
            try:
                evidence = dict(observations=observations, attempts=attempts())
            except httpx.HTTPError:
                evidence = dict(observations=observations, receiver_unavailable=True)
            (directory / 'monitoring-attempts.json').write_text(json.dumps(evidence, indent=2) + '\n')
