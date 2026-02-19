$ErrorActionPreference = 'Stop'

$isDotSourced = $MyInvocation.InvocationName -eq '.'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $isDotSourced) {
    Write-Warning 'Запускайте скрипт через dot-source: . .\scripts\setup.ps1'
}

$venvDir = Join-Path $repoRoot '.venv'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'

if (-not (Test-Path $venvDir)) {
    Write-Host '[i] Создаю .venv ...'
    python -m venv .venv
}

if (-not (Test-Path $activateScript)) {
    throw "Не найден Activate.ps1: $activateScript"
}

. $activateScript
Write-Host "[OK] Активировано: $env:VIRTUAL_ENV"

python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/00_setup_env.py --require-piper-training --torch skip

Write-Host "`n=== Диагностика Piper ==="
python -c "import piper; print('piper module:', piper.__file__)"
python -c "import piper.espeakbridge as b; print('piper.espeakbridge:', b.__file__)"
python -c "import piper; from pathlib import Path; p=Path('third_party/piper1-gpl/src/piper').resolve(); piper.__path__.append(str(p)) if str(p) not in piper.__path__ else None; import piper.train.vits as v; print('piper.train.vits:', v.__file__)"
