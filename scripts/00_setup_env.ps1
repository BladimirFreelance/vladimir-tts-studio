$ErrorActionPreference = 'Stop'

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

$python = Resolve-Python
$probe = Invoke-WithPython -PythonCommand $python -Arguments @('-c', "import sys; print('.'.join(map(str, sys.version_info[:3])))")
if (-not $probe) {
    throw "Не удалось определить версию Python."
}

Write-Host "[i] Использую Python $probe через: $($python -join ' ')"

Invoke-WithPython -PythonCommand $python -Arguments (@('scripts/00_setup_env.py') + $args)
