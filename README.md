# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.

## Quick Start (Windows)

```powershell
git clone <repo-url> && cd vladimir-tts-studio
. .\scripts\00_bootstrap.ps1 --mode training --require-piper-training
python -m app.main train --project <имя> --check
```

> Для запуска обучения после preflight-проверки: `python -m app.main train --project <имя>`.


## Структура зависимостей

- `requirements/base.txt` — базовые зависимости приложения.
- `requirements/runtime.txt` — зависимости для runtime-синтеза (включая `piper-tts`).
- `requirements/train.txt` — база для training-потока; training-модули Piper ставятся через `third_party/piper1-gpl[train]` в `scripts/00_setup_env.py`.

## Что сохранять перед удалением/переустановкой

Если планируете удалить репозиторий, переустановить систему или перенести проект на другой ПК, разделяйте **данные** и **окружение**.

### Сохранять обязательно (данные проекта)

- `data/projects/<name>/recordings/`
- `data/projects/<name>/metadata/`
- `data/projects/<name>/runs/`

При необходимости можно сохранить весь каталог `data/projects/<name>/` целиком.

### Сохранять не нужно (пересоздаётся автоматически)

- `data/projects/<name>/.venv`
- корневой `.venv` (если есть)
- временные файлы/кэш внутри репозитория

После восстановления репозитория просто повторите шаги установки из раздела выше.

## Быстрые команды

### Запуск UI

```powershell
python -m app.main record --project <имя> --port 8765
```

### Диагностика и авто-исправление

```powershell
python scripts/06_doctor.py --project <имя> --auto-fix
```

### Проверка, что модуль обучения доступен

```powershell
python -m training.piper_train_bootstrap --help
```
