#!/usr/bin/env python3
"""Publish measured quality summaries without copying data or model artifacts."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recserve_app.model import file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
args = parser.parse_args()
report = json.loads(args.report.read_text())
for stage in ('validation', 'test'):
    for metrics in report[stage].values():
        metrics.pop('user_ndcg', None)
report['raw_report_sha256'] = file_hash(args.report)
report['raw_report'] = args.report.resolve().relative_to(ROOT).as_posix()
report['release_manifest'] = Path(report['release_manifest']).resolve().relative_to(ROOT).as_posix()
destination = ROOT / 'results' / ('production-quality-' + report['training']['dataset'] + '.json')
destination.write_text(json.dumps(report, indent=2) + '\n')
print(destination)
