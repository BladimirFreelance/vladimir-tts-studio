# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.


## Хранение данных

- Папка `data/` предназначена для рабочих датасетов и должна храниться отдельно от репозитория (например, на отдельном диске/папке с бэкапом).
- В git хранятся только шаблонные `.gitkeep`; реальные записи, проекты и артефакты не коммитьте.

## Быстрая автономная установка на чистом устройстве

1) Установите Python 3.11+ и Git.
2) Установите **eSpeak NG** (это отдельная программа, не pip-пакет).
3) В корне репозитория выполните:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\00_setup_env.ps1
```

Что делает скрипт:
- создаёт/активирует `.venv`;
- устанавливает зависимости проекта;
- ставит runtime/training-компоненты Piper;
- подготавливает базу для дальнейших команд.

> Для полностью автоматического сценария «с нуля» можно использовать one-click:
>
> ```powershell
> powershell -ExecutionPolicy Bypass -File .\scripts\00_one_click.ps1 -Project ddn_vladimir -Text .\data\projects\ddn_vladimir\input_texts\testdata.txt
> ```

## Важные инструменты (без лишнего)

### 1) Запуск UI

Единая панель (подготовка, запись, обучение, экспорт, тест, doctor):

```bash
python -m app.main record --project ddn_vladimir --port 8765
```

### 2) Проверка обучения

Проверка, что training-модули доступны и команда обучения работает:

```bash
python scripts/00_setup_env.py --require-piper-training
python -m training.piper_train_bootstrap --help
```

Запуск обучения:

```bash
python -m app.main train --project ddn_vladimir
```

### 3) Doctor (исправлено и обязательно)

Основная диагностика и авто-исправление:

```bash
python scripts/06_doctor.py --project ddn_vladimir --auto-fix
```

`doctor` проверяет:
- `piper.train`;
- `piper.espeakbridge` (опционально, для runtime-сценариев);
- `espeak-ng`;
- `torch`/GPU;
- manifest и наличие WAV;
- формат аудио (22050 Hz / mono / int16) и длительности.

## Инструменты авто-диагностики и решения проблем

Используйте эти команды в таком порядке:

1. **Полная автоподготовка окружения**
   ```bash
   python scripts/00_setup_env.py --torch auto --require-piper-training
   ```
2. **Глубокая диагностика проекта и авто-исправления**
   ```bash
   python scripts/06_doctor.py --project ddn_vladimir --auto-fix
   ```
3. **Проверка bootstrap обучения**
   ```bash
   python -m training.piper_train_bootstrap --help
   ```
4. **Перезапуск через UI**
   ```bash
   python -m app.main record --project ddn_vladimir --port 8765
   ```

Если нужен runtime-модуль `piper.espeakbridge`, переустановите Piper-пакет в `.venv`:

```bash
./.venv/Scripts/python.exe -m pip install -e "./third_party/piper1-gpl[train]"
```
