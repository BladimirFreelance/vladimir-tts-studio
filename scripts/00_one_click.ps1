param(
    [Parameter(Mandatory = $true)]
    [string]$Project,
    [string]$Text,
    [bool]$UseGPU = $true
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Assert-Command {
    param([string]$Name, [string]$Hint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "$Name не найден. $Hint"
    }
}

function Get-Python312 {
    $candidates = @('py -3.12', 'python')
    foreach ($candidate in $candidates) {
        $parts = $candidate.Split(' ')
        $cmd = $parts[0]
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
            continue
        }

        try {
            $version = & $cmd @($parts[1..($parts.Length - 1)]) -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($LASTEXITCODE -eq 0 -and $version -eq '3.12') {
                return @($cmd) + @($parts[1..($parts.Length - 1)])
            }
        }
        catch {
            continue
        }
    }

    Write-Error 'Python 3.12 не найден. Установите Python 3.12 и перезапустите скрипт.'
}

function Ensure-Espeak {
    $espeak = (Get-Command espeak-ng -ErrorAction SilentlyContinue)
    if ($espeak) {
        Write-Host "[ok] eSpeak NG найден: $($espeak.Source)"
        return
    }

    $defaultPath = 'C:\Program Files\eSpeak NG\espeak-ng.exe'
    if (Test-Path $defaultPath) {
        $env:PATH = "$(Split-Path $defaultPath);$env:PATH"
        Write-Host "[ok] eSpeak NG найден в стандартном пути: $defaultPath"
        return
    }

    Write-Error "eSpeak NG не найден. Установите eSpeak NG (ожидаемый путь: '$defaultPath'), затем перезапустите скрипт."
}

Assert-Command -Name 'git' -Hint 'Установите Git for Windows и добавьте git.exe в PATH.'
$python = Get-Python312

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

if (-not (Test-Path '.venv')) {
    & $python -m venv .venv
}

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error 'Не найден python.exe внутри .venv. Удалите .venv и запустите снова.'
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -e .

$depsRoot = Join-Path $repoRoot 'third_party'
$piperDir = Join-Path $depsRoot 'piper1-gpl'
$piperRepo = 'https://github.com/OHF-Voice/piper1-gpl.git'
$piperRef = 'origin/main'

if (-not (Test-Path $piperDir)) {
    New-Item -ItemType Directory -Path $depsRoot -Force | Out-Null
    git clone $piperRepo $piperDir
}

git -C $piperDir fetch --all --tags
git -C $piperDir checkout $piperRef

& $venvPython -m pip uninstall -y piper-tts
$piperEditableSpec = "${piperDir}[train]"
& $venvPython -m pip install -e $piperEditableSpec

& $venvPython -c "import importlib.util as u; assert u.find_spec('piper.train.vits') is not None"
& $venvPython -m training.piper_train_bootstrap --help | Out-Null

Ensure-Espeak

if ($Text) {
    & $venvPython scripts/01_prepare_dataset.py --project $Project --text $Text
    $manifest = Join-Path $repoRoot "data/projects/$Project/metadata/train.csv"
    if (-not (Test-Path $manifest)) {
        Write-Error "После prepare не найден manifest: $manifest"
    }
}

& $venvPython -m app.main doctor --project $Project

if (-not $UseGPU) {
    $env:CUDA_VISIBLE_DEVICES = ''
}

& $venvPython -m app.main train --project $Project
