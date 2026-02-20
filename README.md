# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.

## Quick Start (Windows)

```powershell
. .\scripts\00_bootstrap.ps1 --mode training --require-piper-training
python -m app.main train --check
```

> Для запуска обучения после preflight-проверки: `python -m app.main train`.

По умолчанию имя проекта берётся из имени репозитория (`vladimir-tts-studio`).

## Автопоиск файлов

- Для `prepare` теперь можно не указывать `--text`.
- Приложение автоматически ищет первый подходящий файл `.txt/.csv/.tsv` в:
  1. `data/input_texts/`
  2. `data/`
  3. `dataset/`
  4. корне репозитория

Пример:

```powershell
python -m app.main prepare
```

Если нужно использовать конкретный файл, можно передать его явно:

```powershell
python -m app.main prepare --text data/input_texts/news.txt
```

## Структура зависимостей

- `requirements/base.txt` — базовые зависимости приложения.
- `requirements/runtime.txt` — зависимости для runtime-синтеза (включая `piper-tts`).
- `requirements/train.txt` — база для training-потока; training-модули Piper ставятся через `third_party/piper1-gpl[train]` в `scripts/00_setup_env.py`.

## Что сохранять перед удалением/переустановкой

Если планируете удалить репозиторий, переустановить систему или перенести проект на другой ПК, разделяйте **данные** и **окружение**.

### Сохранять обязательно (данные проекта)

- `data/projects/vladimir-tts-studio/recordings/`
- `data/projects/vladimir-tts-studio/metadata/`
- `data/projects/vladimir-tts-studio/runs/`

При необходимости можно сохранить весь каталог `data/projects/vladimir-tts-studio/` целиком.

### Сохранять не нужно (пересоздаётся автоматически)

- `data/projects/vladimir-tts-studio/.venv`
- корневой `.venv` (если есть)
- временные файлы/кэш внутри репозитория

После восстановления репозитория просто повторите шаги установки из раздела выше.

## Быстрые команды

### Запуск UI

```powershell
python -m app.main record --port 8765
```

### Диагностика и авто-исправление

```powershell
python scripts/06_doctor.py --auto-fix
```

### Проверка, что модуль обучения доступен

```powershell
python -m training.piper_train_bootstrap --help
```
