$ErrorActionPreference = 'Stop'

$isDotSourced = $MyInvocation.InvocationName -eq '.'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Test-PythonCommand {
    param([string[]]$PythonCommand)

    $exe = $PythonCommand[0]
    $prefix = @()
    if ($PythonCommand.Length -gt 1) {
        $prefix = $PythonCommand[1..($PythonCommand.Length - 1)]
    }

    $probe = & $exe @prefix -c "import sys; print('.'.join(map(str, sys.version_info[:3]))); sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $probe) {
        return $probe.Trim()
    }

    return $null
}

function Find-WorkingPython {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source -and -not $pythonCmd.Source.Contains('WindowsApps')) {
        $version = Test-PythonCommand -PythonCommand @('python')
        if ($version) {
            return @{ Command = @('python'); Version = $version }
        }
    }

    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        foreach ($candidate in @('-3.12', '-3.11', '-3')) {
            $version = Test-PythonCommand -PythonCommand @('py', $candidate)
            if ($version) {
                return @{ Command = @('py', $candidate); Version = $version }
            }
        }
    }

    $directCandidates = @(
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($candidatePath in $directCandidates) {
        if (-not (Test-Path $candidatePath)) {
            continue
        }
        $version = Test-PythonCommand -PythonCommand @($candidatePath)
        if ($version) {
            return @{ Command = @($candidatePath); Version = $version }
        }
    }

    return $null
}

function Resolve-Python {
    $resolved = Find-WorkingPython
    if ($resolved) {
        return $resolved
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host '[i] Python >= 3.11 не найден. Пробую установить Python 3.12 через winget (scope=user)...'
        & winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось установить Python через winget (код $LASTEXITCODE)."
        }

        $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
        $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
        $env:Path = "$userPath;$machinePath"

        $resolved = Find-WorkingPython
        if ($resolved) {
            return $resolved
        }
    }

    throw "Не найден рабочий Python (>=3.11). Установите Python 3.12 с https://www.python.org/downloads/windows/ (включите Add python.exe to PATH), либо установите через winget: winget install -e --id Python.Python.3.12"
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
    $pythonInfo = Resolve-Python
    $python = $pythonInfo.Command
    Write-Host "[i] Использую Python $($pythonInfo.Version) через: $($python -join ' ')"
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
