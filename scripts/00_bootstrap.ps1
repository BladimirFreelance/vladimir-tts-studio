param(
    [switch]$Install
)

$ErrorActionPreference = 'Stop'

$isDotSourced = $MyInvocation.InvocationName -eq '.'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvDir = Join-Path $repoRoot '.venv'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'

if (-not (Test-Path $venvDir)) {
    Write-Host '[i] .venv не найден, создаю виртуальное окружение...'

    $created = $false
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            & py -3.12 -m venv .venv
            $created = $true
            Write-Host '[OK] .venv создан через py -3.12'
        }
        catch {
            Write-Host '[WARN] py -3.12 недоступен, пробую python -m venv .venv'
        }
    }

    if (-not $created) {
        & python -m venv .venv
        Write-Host '[OK] .venv создан через python -m venv'
    }
}
else {
    Write-Host '[i] .venv уже существует.'
}

if (-not (Test-Path $activateScript)) {
    throw "Не найден скрипт активации: $activateScript"
}

if (-not $isDotSourced) {
    Write-Warning 'Скрипт запущен без dot-sourcing: активация не сохранится в текущей сессии.'
    Write-Host 'Используйте именно так:'
    Write-Host '. .\scripts\00_bootstrap.ps1'
}

. $activateScript
Write-Host "[OK] Активировано окружение: $env:VIRTUAL_ENV"

python -c "import site,sys; print('sys.executable =', sys.executable); print('site-packages =', site.getsitepackages())"
pip --version

if ($Install) {
    Write-Host '[i] Запускаю установку зависимостей с обязательной проверкой training...'
    python scripts/00_setup_env.py --require-piper-training
}

Write-Host "`nПроверка активного окружения:"
Write-Host "1) python -c \"import sys; print(sys.executable)\""
Write-Host "2) python -m pip --version"
Write-Host "3) python -m app.main doctor --project <PROJECT_NAME>"
