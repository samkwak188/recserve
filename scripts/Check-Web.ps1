param(
    [ValidateSet('build', 'install-browser', 'test')][string]$Stage = 'build',
    [string]$BaseURL = '',
    [string]$FixturePath = '',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Join-Path (Split-Path -Parent $PSScriptRoot) 'web')
try {
    if ($Stage -eq 'build') {
        & npm.cmd run types
        if ($LASTEXITCODE -ne 0) { throw 'API type generation failed' }
        & npm.cmd run build
    } elseif ($Stage -eq 'install-browser') {
        & npx.cmd playwright install chromium
    } else {
        $env:PLAYWRIGHT_BASE_URL = $BaseURL
        $env:BROWSER_FIXTURE = $FixturePath
        $env:BROWSER_OUTPUT_DIR = $OutputDirectory
        & npm.cmd test
    }
    if ($LASTEXITCODE -ne 0) { throw "Web stage $Stage failed" }
} finally { Pop-Location }
