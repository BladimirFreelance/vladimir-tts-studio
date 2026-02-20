# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.

## Установка и первый запуск (Windows)

### 1) Клонируйте репозиторий

```powershell
git clone <repo-url>
cd vladimir-tts-studio
```

### 2) Запустите bootstrap-скрипт

```powershell
.\scripts\00_bootstrap.ps1
```

### 3) Подготовьте окружение для обучения Piper

```powershell
python scripts/00_setup_env.py --require-piper-training
```

### 4) Запустите обучение

```powershell
.\scripts\run.ps1 train --project <имя>
```

> Новый рекомендуемый поток установки: `00_bootstrap.ps1` → `00_setup_env.py --require-piper-training` → `run.ps1 train --project ...`.

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
