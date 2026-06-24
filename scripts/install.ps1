# Cross-platform install script for fiber-tracer (Windows PowerShell).

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Test-PythonVersion {
    param([string]$Command)
    try {
        $versionString = & $Command -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        $parts = $versionString -split "\."
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        return ($major -ge 3 -and ($major -gt 3 -or $minor -ge 10))
    } catch {
        return $false
    }
}

$PythonCmd = $null
foreach ($cmd in @("python3.12", "python3.11", "python3.10", "python3")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        if (Test-PythonVersion -Command $cmd) {
            $PythonCmd = $cmd
            break
        }
    }
}

if (-not $PythonCmd) {
    Write-Error "Python >=3.10 is required but was not found. Please install Python 3.10, 3.11, or 3.12 and try again."
    exit 1
}

Write-Host "Using Python: $PythonCmd (& $PythonCmd --version)"

if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Host "Bun not found. Installing Bun..."
    powershell -c "irm bun.sh/install.ps1|iex"
    $env:PATH = "$env:USERPROFILE\.bun\bin;$env:PATH"
} else {
    Write-Host "Bun is already installed: $(Get-Command bun | Select-Object -ExpandProperty Source)"
}

Set-Location $RepoRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment with $PythonCmd..."
    & $PythonCmd -m venv .venv
}

& .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

Set-Location tui
bun install

Write-Host ""
Write-Host "Installation complete."
Write-Host "Activate the virtual environment with: .venv\Scripts\Activate.ps1"
