#!/usr/bin/env python3
"""Read hosted CI status for this checkout; never treat queued jobs as passed."""
import argparse
import json
import pathlib
import subprocess
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sha')
args = parser.parse_args()
sha = args.sha or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
def get(url):
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'recserve-validation'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)
root = 'https://api.github.com/repos/samkwak188/recserve'
runs = get(root + '/actions/runs?head_sha=' + sha)['workflow_runs']
summary = []
for run in runs:
    jobs = get(run['jobs_url'] + '?per_page=100')['jobs']
    summary.append(dict(id=run['id'], url=run['html_url'], status=run['status'], conclusion=run['conclusion'],
        jobs=[dict(name=j['name'], status=j['status'], conclusion=j['conclusion'], url=j['html_url'],
                   failed_steps=[s['name'] for s in j.get('steps', []) if s['conclusion'] == 'failure']) for j in jobs]))
result = dict(sha=sha, runs=summary)
path = pathlib.Path('.cache/logs/hosted-ci.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2)+'\n')
path.with_name('hosted-ci-' + sha[:8] + '.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(sha=sha, runs=[dict(id=r['id'], status=r['status'], conclusion=r['conclusion'],
    passed=sum(j['conclusion'] == 'success' for j in r['jobs']),
    pending=[j['name'] for j in r['jobs'] if j['status'] != 'completed'],
    failed=[j for j in r['jobs'] if j['conclusion'] not in (None, 'success', 'skipped')]) for r in summary])))
