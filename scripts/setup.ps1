$ErrorActionPreference = 'Stop'

$isDotSourced = $MyInvocation.InvocationName -eq '.'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Resolve-Python {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -and -not $pythonCmd.Source.Contains('WindowsApps')) {
        return @('python')
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        foreach ($candidate in @('-3.12', '-3.11', '-3')) {
            $probe = & py $candidate -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) {
                return @('py', $candidate)
            }
        }
    }

    throw "Не найден рабочий Python (>=3.11). Установите Python 3.11/3.12 и добавьте его в PATH (или установите py launcher)."
}

function Invoke-WithPython {
    param(
        [string[]]$PythonCommand,
        [string[]]$Arguments
    )

    $exe = $PythonCommand[0]
    $prefix = @()
    if ($PythonCommand.Length -gt 1) {
        $prefix = $PythonCommand[1..($PythonCommand.Length - 1)]
    }

    & $exe @prefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Команда '$($PythonCommand -join ' ') $($Arguments -join ' ')' завершилась с кодом $LASTEXITCODE. Если видите только 'Python', отключите App execution aliases для python.exe/python3.exe в Windows Settings и повторите запуск."
    }
}

if (-not $isDotSourced) {
    Write-Warning 'Запускайте скрипт через dot-source: . .\scripts\setup.ps1'
}

$venvDir = Join-Path $repoRoot '.venv'
$activateScript = Join-Path $venvDir 'Scripts\Activate.ps1'

if (-not (Test-Path $venvDir)) {
    Write-Host '[i] Создаю .venv ...'
    $python = Resolve-Python
    Invoke-WithPython -PythonCommand $python -Arguments @('-m', 'venv', '.venv')
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
