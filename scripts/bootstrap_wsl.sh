#!/usr/bin/env bash
# Run as root in Ubuntu WSL; install development components, never Linux drivers.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .cache/logs
exec >.cache/logs/bootstrap-wsl.log 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends cmake ninja-build python3-venv python3-dev librdkafka-dev curl ca-certificates
curl --fail --location --retry 3 https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -o .cache/cuda-keyring.deb
dpkg -i .cache/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends cuda-nvcc-13-1 cuda-cudart-dev-13-1 cuda-cccl-13-1 libcublas-dev-13-1 cuda-sanitizer-13-1
cmake --version
/usr/local/cuda-13.1/bin/nvcc --version
printf '\nBOOTSTRAP_OK\n'
