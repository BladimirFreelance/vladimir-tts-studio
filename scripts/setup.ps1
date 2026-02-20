$ErrorActionPreference = 'Stop'

$isDotSourced = $MyInvocation.InvocationName -eq '.'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $isDotSourced) {
    Write-Warning 'Рекомендуется dot-source: . .\scripts\setup.ps1 (так .venv останется активным в текущей сессии).'
}

$venvDir = Join-Path $repoRoot '.venv'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'

if (-not (Test-Path $venvDir)) {
    Write-Host '[i] .venv не найден. Запускаю bootstrap setup...'
    & powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot 'scripts\00_setup_env.ps1') --venv .venv --require-piper-training --torch skip
    if ($LASTEXITCODE -ne 0) {
        throw "Bootstrap setup завершился с кодом $LASTEXITCODE"
    }
}

if (-not (Test-Path $activateScript)) {
    throw "Не найден Activate.ps1: $activateScript"
}

. $activateScript
Write-Host "[OK] Активировано: $env:VIRTUAL_ENV"

python -c "import sys; print('python:', sys.executable)"
python -m pip --version

Write-Host "`n=== Диагностика Piper ==="
python -c "import piper; print('piper module:', piper.__file__)"
python -c "import piper; from pathlib import Path; p=Path('third_party/piper1-gpl/src/piper').resolve(); piper.__path__.append(str(p)) if str(p) not in piper.__path__ else None; import piper.train.vits as v; print('piper.train.vits:', v.__file__)"
