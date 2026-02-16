# vladimir-piper-voice-lab

Полностью Python-проект для Windows 10/11 (Python 3.11/3.12), который ведёт пользователя по этапам:

1. Подготовка датасета из одного `.txt`
2. Запись в браузере (локальная студия)
3. Обучение Piper
4. Экспорт ONNX
5. Тест синтеза
6. Диагностика (`doctor`)

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
python scripts/05_test_voice.py --model voices_out/ru_RU-ddn_vladimir-medium.onnx --text "Привет!" --out test.wav
python scripts/06_doctor.py --project ddn_vladimir --auto-fix
```

## Структура проекта

- `app/main.py` — единый CLI (`prepare / record / train / export / test / doctor`)
- `app/doctor.py` — проверки окружения и датасета
- `dataset/` — очистка текста, сегментация, manifest, audio-tools
- `studio/` — локальный FastAPI сервер и web studio
- `training/` — обёртки обучения/экспорта/инференса
- `scripts/01..06` — отдельные удобные точки входа

## Формат датасета

- Manifest: `data/projects/<project>/metadata/train.csv`
- Разделитель строго `|` (`audio|text`), UTF-8, **без заголовка**
- `audio` путь относительный к корню проекта (`recordings/wav_22050/<id>.wav`)

## Doctor checks

`python scripts/06_doctor.py --project <name> --auto-fix`

Проверяет:

- `import piper.espeakbridge`
- наличие `espeak-ng`
- доступность `torch`/CUDA
- каждую строку manifest
- существование WAV
- формат WAV (22050/mono/int16)
- длительность (1–12 сек)

При `--auto-fix` выполняет конвертацию WAV через `ffmpeg`, а если его нет — Python fallback.

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

Вы записали 0 файлов или manifest указывает на несуществующие пути. Запустите:

```bash
python scripts/06_doctor.py --project <project> --auto-fix
```

### `missing wav`

Строки есть, но нет файлов в `recordings/wav_22050`. Перезапишите пропущенные строки в web-студии.

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
