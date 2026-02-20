$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvDir = Join-Path $repoRoot '.venv'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'
$setupScript = Join-Path $repoRoot 'scripts\00_setup_env.py'

$isDotSourced = $MyInvocation.InvocationName -eq '.'

Write-Host '[i] Для активации в текущей PowerShell-сессии запускайте скрипт через dot-sourcing:'
Write-Host ("    . '{0}'" -f (Join-Path $repoRoot 'scripts\00_bootstrap.ps1'))

Push-Location $repoRoot
try {
    if (-not (Test-Path $venvDir)) {
        Write-Host '[i] .venv не найден, создаю виртуальное окружение...'

        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            & py -3.12 -m venv $venvDir
            if ($LASTEXITCODE -eq 0) {
                Write-Host '[OK] .venv создан через py -3.12'
            }
            else {
                throw 'Не удалось создать .venv через py -3.12.'
            }
        }
        else {
            & python -m venv $venvDir
            if ($LASTEXITCODE -eq 0) {
                Write-Host '[OK] .venv создан через python -m venv'
            }
            else {
                throw 'Не удалось создать .venv через python -m venv.'
            }
        }
    }
    else {
        Write-Host '[i] .venv уже существует.'
    }

    if (-not (Test-Path $activateScript)) {
        throw "Не найден скрипт активации: $activateScript"
    }

    if (-not (Test-Path $setupScript)) {
        throw "Не найден скрипт настройки окружения: $setupScript"
    }

    if (-not $isDotSourced) {
        Write-Warning 'Скрипт запущен без dot-sourcing: активация не сохранится в текущей сессии.'
    }

    . $activateScript
    Write-Host "[OK] Активировано окружение: $env:VIRTUAL_ENV"

    python -c "import sys; print('sys.executable =', sys.executable)"

    Write-Host '[i] Запускаю scripts/00_setup_env.py...'
    python $setupScript @args
}
finally {
    Pop-Location
}
