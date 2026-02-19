# vladimir-piper-voice-lab

Полностью Python-проект для Windows 10/11 (Python 3.11/3.12), который ведёт пользователя по этапам:

1. Подготовка датасета из одного `.txt`
2. Запись в браузере (локальная студия)
3. Обучение Piper
4. Экспорт ONNX
5. Тест синтеза
6. Диагностика (`doctor`)

## Windows prerequisites

- Python 3.12
- Git for Windows
- **eSpeak NG (exe)**: установите отдельно, это не Python-пакет.

Проверка:

```powershell
espeak-ng --version
# или
& "C:\Program Files\eSpeak NG\espeak-ng.exe" --version
```

## Windows Quick Start (важно!)

После свежего clone запускайте проект только из `.venv`, чтобы не попасть в случайно активное окружение (`.venv1` и т.п.). `(.venv1)` не используйте.

### После clone: один шаг setup

```powershell
. .\scripts\setup.ps1
```

Первая точка (`.`) **обязательна**: это dot-sourcing, он создаёт/активирует `.venv` в текущей сессии, ставит зависимости и делает проверки `piper.espeakbridge` + `piper.train.vits`.

Запуск обучения после setup:

```powershell
python -m app.main train --project ddn_vladimir
```

## Авто-установка зависимостей (рекомендуется для PyCharm Terminal)

Теперь можно запустить один скрипт, который создаст `.venv`, обновит `pip`, установит зависимости проекта, PyTorch, `piper-tts` (runtime) и Piper training-модули:

```bash
python scripts/00_setup_env.py
```

Если в PowerShell команда `python scripts/00_setup_env.py` возвращает только `Python` (частый случай с alias из `WindowsApps`), используйте обёртку:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\00_setup_env.ps1
```

Она подберёт рабочий интерпретатор (`python` или `py -3.12`) и передаст все аргументы в `00_setup_env.py`.

Полезные варианты:

```bash
python scripts/00_setup_env.py --torch auto      # авто-детект GPU/драйверов и подбор сборки
python scripts/00_setup_env.py --torch cu124     # CUDA 12.4+ сборка PyTorch
python scripts/00_setup_env.py --torch cu121     # CUDA 12.1 сборка PyTorch
python scripts/00_setup_env.py --torch directml  # AMD/Intel GPU на Windows (без WSL)
python scripts/00_setup_env.py --torch skip      # пропустить torch
python scripts/00_setup_env.py --no-venv         # ставить в текущее окружение
python scripts/00_setup_env.py --without-piper-training  # пропустить установку Piper training-модулей
python scripts/00_setup_env.py --with-piper-training     # явное включение установки training (алиас)
python scripts/00_setup_env.py --require-piper-training   # упасть с ошибкой, если training-модули не установились
```

> При `--require-piper-training` setup подготавливает `third_party/piper1-gpl` и проверяет `piper.train.vits` через bootstrap без замены runtime-пакета `piper-tts`.

## Обучение на GPU в Windows (без WSL)

Скрипт `scripts/00_setup_env.py` теперь автоматически определяет конфигурацию ПК и ставит максимально совместимый стек:

- **NVIDIA GPU**: читает `nvidia-smi` и по умолчанию ставит CUDA-сборку (`cu124`/`cu121`, в зависимости от версии CUDA).
- **AMD/Intel GPU (Windows)**: ставит CPU-колёса PyTorch + `torch-directml` для запуска обучения через DirectML.
- **Fallback**: если CUDA-колёса не ставятся, на Windows пробует DirectML и затем CPU, на других ОС — CPU.

Для ручного контроля можно явно указать `--torch`.

## One-click (чистый клон, Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\00_one_click.ps1 -Project ddn_vladimir -Text .\data\projects\ddn_vladimir\input_texts\testdata.txt
```

Скрипт создаёт `.venv`, ставит проект, проверяет `piper.train` и `eSpeak NG`, запускает `doctor`, затем `train`.

## Быстрый старт (PyCharm / обычный `python`)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Положите текст в `data/input_texts/news.txt` (UTF-8).

```bash
python scripts/01_prepare_dataset.py --text data/input_texts/news.txt --project ddn_vladimir
python scripts/02_run_studio.py --project ddn_vladimir --port 8765
python scripts/03_train.py --project ddn_vladimir --epochs 50
python scripts/04_export_onnx.py --project ddn_vladimir
python scripts/05_test_voice.py --model voices_out/ru_RU-ddn_vladimir-medium.onnx --config voices_out/ru_RU-ddn_vladimir-medium.onnx.json --text "Привет!"
python scripts/06_doctor.py --project ddn_vladimir --auto-fix
```

## Графическая оболочка (единая панель)

Теперь `record` открывает единую GUI-панель, где доступны **все этапы** пайплайна: подготовка, запись, обучение, экспорт, тест синтеза и doctor.

```bash
python -m app.main record --project ddn_vladimir --port 8765
```

В интерфейсе:
- блок статуса и фоновых задач;
- форма подготовки датасета по пути к `.txt`;
- студия записи фраз;
- запуск обучения и экспорта ONNX;
- тест синтеза в WAV;
- doctor-проверка с `auto-fix`.

## Структура проекта

- `app/main.py` — единый CLI (`prepare / record / train / export / test / doctor`)
- `app/doctor.py` — проверки окружения и датасета
- `dataset/` — очистка текста, сегментация, manifest, audio-tools
- `studio/` — локальный FastAPI сервер и web studio
- `training/` — обёртки обучения/экспорта/инференса
- `scripts/01..06` — отдельные удобные точки входа

## Формат датасета

- Manifest: `data/projects/PROJECT_NAME/metadata/train.csv`
- Разделитель строго `|` (`audio|text`), UTF-8, **без заголовка**
- `audio` = только имя WAV-файла (`ddn_vladimir_00001.wav`)

## Doctor checks

`python scripts/06_doctor.py --project PROJECT_NAME --auto-fix`

Проверяет:

- `import piper.espeakbridge`
- доступность модуля обучения (`piper.train`)
- наличие `espeak-ng`
- доступность `torch`/CUDA
- каждую строку manifest
- существование WAV
- формат WAV (22050/mono/int16)
- длительность (1–12 сек)

`prepare` и `doctor --auto-fix` приводят найденные записи к формату 22050 Hz / mono / int16 (через `ffmpeg`, а если его нет — Python fallback).

## Piper/espeakbridge

- Runtime/инференс ожидает установленный `piper-tts`.
- Если `doctor` пишет `ImportError: piper.espeakbridge`:

```bash
pip install piper-tts
```

## Опциональные улучшения в проекте

- Анализ качества записи (`dataset/audio_tools.py::analyze_quality`) — clipping/noise/rms
- Авто-конвертация аудио в правильный формат
- Можно сгенерировать короткий тест из 20 фраз и сравнивать прогресс чекпойнтов

## Типовые ошибки

### `0 utterances`

Вы записали 0 файлов или manifest указывает на несуществующие пути. Если вы запустили doctor до `prepare`, сначала создайте проект и manifest.

```bash
python scripts/06_doctor.py --project PROJECT_NAME --auto-fix
```

### `missing wav`

Строки есть, но нет файлов в `recordings/wav_22050`. Перезапишите пропущенные строки в web-студии.

### `ModuleNotFoundError: No module named 'piper.train'`

Если окружение создавалось не через `scripts/00_setup_env.py`, training-модули могли не установиться. Варианты:

```bash
# 1) указать явную команду обучения (PowerShell)
$env:PIPER_TRAIN_CMD="python -m training.piper_train_bootstrap"

# 2) или переустановить окружение (по умолчанию training ставится автоматически)
python scripts/00_setup_env.py

# 3) если нужно, чтобы setup обязательно падал при отсутствии training-модулей
python scripts/00_setup_env.py --require-piper-training
```

Smoke test после setup:

```bash
python scripts/00_setup_env.py --require-piper-training
python -c "import importlib.util as u; print(u.find_spec('piper.train.vits'))"
python -m training.piper_train_bootstrap --help
```

После этого снова запустите обучение из студии.

### Невнятный/"бормочущий" голос

- мало данных (нужно больше минут/часов)
- шум или клиппинг в записи
- разный микрофон/громкость
- слишком длинные реплики

## Рекомендации по данным

- минимум 30–60 минут речи, лучше 2+ часа
- один и тот же микрофон и расстояние
- тихое помещение
- без перегруза (clipping)
- короткие фразы (примерно 3–10 сек)
