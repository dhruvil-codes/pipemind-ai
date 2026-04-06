# Pipeline Debugger — Start Script
# Usage: .\start.ps1
# Or with API key: .\start.ps1 -ApiKey "sk-..."

param(
    [string]$ApiKey = $env:HF_TOKEN
)

$env:PYTHONPATH = "C:\Users\Dhruvil\Downloads"

if ($ApiKey) {
    $env:HF_TOKEN = $ApiKey
    Write-Host "✓ HF_TOKEN set" -ForegroundColor Green
} else {
    Write-Host "⚠ No API key — uploads will use direct execution (no LLM iteration)" -ForegroundColor Yellow
}

Write-Host "Starting Pipeline Debugger on http://localhost:8000 ..." -ForegroundColor Cyan
uvicorn pipeline_debugger_env.server.app:app --host 0.0.0.0 --port 8000 --reload
