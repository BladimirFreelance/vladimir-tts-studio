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
        return @('py', '-3.12')
    }

    throw "Не найден рабочий Python. Установите Python 3.12 и добавьте его в PATH (или установите py launcher)."
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
}

$python = Resolve-Python
$probe = Invoke-WithPython -PythonCommand $python -Arguments @('-c', "import sys; print('.'.join(map(str, sys.version_info[:3])))")
if (-not $probe) {
    throw "Не удалось определить версию Python."
}

Write-Host "[i] Использую Python $probe через: $($python -join ' ')"

Invoke-WithPython -PythonCommand $python -Arguments (@('scripts/00_setup_env.py') + $args)
