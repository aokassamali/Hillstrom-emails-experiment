param(
  [string]$ConfigPath = "config\config.yaml",
  [string]$OutputDir = "outputs"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv is not installed or not on PATH."
  exit 1
}

if (-not (Test-Path .venv)) {
  uv venv
}

uv pip install -e .
uv run hillstrom-doctor
uv run python -m run_analysis --config $ConfigPath --output $OutputDir
