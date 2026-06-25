# Cross-platform verification script for fiber-tracer (Windows PowerShell).

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".venv")) {
    Write-Error ".venv not found. Please run scripts\install.ps1 first."
    exit 1
}

& .venv\Scripts\Activate.ps1

# Ensure freshly-installed Bun is discoverable in this session.
$BunBin = "$env:USERPROFILE\.bun\bin"
if (-not (Get-Command bun -ErrorAction SilentlyContinue) -and (Test-Path $BunBin)) {
    $env:PATH = "$BunBin;$env:PATH"
}

Write-Host "Running fiber-tracer --version..."
fiber-tracer --version

Write-Host ""
Write-Host "Running pytest..."
pytest

Write-Host ""
Write-Host "Running TUI typecheck and tests..."
Set-Location tui
bun run typecheck
bun test

Write-Host ""
Write-Host "All verification checks passed."
