# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.

## Quick Start (Windows)

```powershell
. .\scripts\00_bootstrap.ps1 --mode training --require-piper-training
python -m app.main train --check --project ddn_vladimir
```

> Для запуска обучения после preflight-проверки: `python -m app.main train --project ddn_vladimir`.

Если `--project` не указан, CLI выбирает проект автоматически:
- если в `data/projects/` ровно один каталог — используется он;
- иначе, если есть `ddn_vladimir` — используется `ddn_vladimir`;
- иначе используется имя репозитория (`vladimir-tts-studio`) и выводится подсказка указать `--project`.

## Runbook after clean clone (Windows, training)

1. `cd` в репозиторий.
2. Выполните bootstrap в training-режиме:

```powershell
. .\scripts\00_bootstrap.ps1 --mode training --require-piper-training
```

3. Проверьте training CLI через bootstrap:

```powershell
python -m training.piper_train_bootstrap --help
```

4. Подготовьте проект (если нет manifest):

```powershell
python -m app.main prepare --project ddn_vladimir
```

5. Запустите диагностику и авто-исправление:

```powershell
python scripts/06_doctor.py --project ddn_vladimir --auto-fix
```

6. Зафиксируйте GPU для процесса обучения:

```powershell
$env:CUDA_DEVICE_ORDER="PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES="1"
```

7. Запустите обучение:

```powershell
python -m app.main train --project ddn_vladimir
```

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
- `requirements/train.txt` — training-зависимости + `piper-tts` wheel.
- `third_party/piper1-gpl` — исходники `piper.train`, подключаются через `training.piper_train_bootstrap` (без `pip install -e third_party/piper1-gpl`).

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


## Если data хранится отдельно

Можно просто скопировать каталог `data/` в чистый клон репозитория и сразу запускать:

```powershell
. .\scripts\00_bootstrap.ps1 --mode training --require-piper-training
python -m app.main train --project ddn_vladimir
```

Bootstrap установит runtime `piper-tts` и подключит training-модули из `third_party/piper1-gpl` автоматически.

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
