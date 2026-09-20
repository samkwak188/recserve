#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/run_logged.py --name gpu-campaign --timeout 7200 -- python3 scripts/gpu_campaign.py "$@"
