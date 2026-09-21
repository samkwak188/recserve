$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
try {
    $listener.Start()
    Write-Output $listener.LocalEndpoint.Port
} finally { $listener.Stop() }
