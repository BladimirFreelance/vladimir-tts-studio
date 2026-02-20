# vladimir-tts-studio

`vladimir-tts-studio` — локальный пайплайн для подготовки датасета, записи, обучения Piper, экспорта ONNX и проверки окружения на Windows.


## Quick Start (Windows)

### 1) Клонируйте репозиторий

```powershell
git clone <repo-url>
cd vladimir-tts-studio
```

### 2) Запустите bootstrap-скрипт

```powershell
.\scripts\00_bootstrap.ps1
```

### 3) Подготовьте Python-окружение и training-компоненты

```powershell
python scripts/00_setup_env.py --require-piper-training
```

### 4) Запустите обучение

Вариант A:

```powershell
python scripts/03_train_one_click.py --project <имя>
```

Вариант B:

```powershell
.\scripts\run.ps1 train --project <имя>
```

## Таблица зависимостей (Windows)

| Компонент | Обязательно | Примечание |
| --- | --- | --- |
| Python 3.12 | Да | Рекомендуется установка из официального дистрибутива Python. |
| Git | Да | Нужен для клонирования и обновления репозитория. |
| eSpeak NG | Да | Установите официальный Windows-инсталлятор: https://github.com/espeak-ng/espeak-ng/releases |
| ffmpeg | Да | Нужен для обработки аудио в пайплайне. |

## Сохранение данных и бэкап

Для переноса проекта на другой ПК или резервного копирования достаточно сохранить папки конкретного проекта:

- `data/projects/<name>/recordings/`
- `data/projects/<name>/metadata/`
- `data/projects/<name>/runs/`

`data/projects/<name>/.venv` сохранять не нужно: виртуальное окружение пересоздаётся командами bootstrap/setup.

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
python -m app.main train --project ddn_vladimir --gpu-name "3060"
# или принудительно CPU
python -m app.main train --project ddn_vladimir --force_cpu
```

One-click обучение (doctor + автоподбор GPU/CPU + train + опциональный export/test):

```bash
python scripts/03_train_one_click.py --project ddn_vladimir --gpu-name "3060"
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
