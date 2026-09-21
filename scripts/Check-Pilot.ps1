# Run from Windows; execution and complete logs remain inside the repository.
param([string]$Distribution = "Ubuntu")
$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & wsl -d $Distribution -- bash scripts/check_pilot.sh
    if ($LASTEXITCODE -ne 0) { throw "Pilot validation failed; inspect .cache/logs/pilot-*.log" }
} finally {
    Pop-Location
}
