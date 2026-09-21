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
        $targetUri = [Uri]$BaseURL
        $reachable = $false
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            $probe = [System.Net.Sockets.TcpClient]::new()
            try {
                $connection = $probe.ConnectAsync('127.0.0.1', $targetUri.Port)
                if ($connection.Wait(500) -and $probe.Connected) { $reachable = $true; break }
            } catch {
                # WSL may become healthy before Windows loopback forwarding is ready.
            } finally { $probe.Dispose() }
            Start-Sleep -Milliseconds 250
        }
        if (-not $reachable) { throw 'The HTTPS fixture is not reachable from Windows; no browser tests were run.' }
        $env:PLAYWRIGHT_BASE_URL = $BaseURL
        $env:BROWSER_FIXTURE = $FixturePath
        $env:BROWSER_OUTPUT_DIR = $OutputDirectory
        & npm.cmd test
    }
    if ($LASTEXITCODE -ne 0) { throw "Web stage $Stage failed" }
} finally { Pop-Location }
