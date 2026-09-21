param(
    [ValidateSet('fast', 'integration', 'browser', 'quality', 'operations', 'qualification', 'release-check')]
    [string]$Stage = 'fast',
    [string]$Distribution = 'Ubuntu',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runnerArgs = @('-d', $Distribution, '--', '.cache/venv-production/bin/python',
                    'scripts/production_runner.py', '--stage', $Stage)
    if ($Resume) { $runnerArgs += '--resume' }
    & wsl @runnerArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed; inspect its .cache/production/runs receipt." }
} finally { Pop-Location }
