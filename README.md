# vladimir-tts-studio

`vladimir-tts-studio` — локальный проект для создания собственного TTS-голоса на базе Piper. Репозиторий объединяет полный цикл работы с голосовой моделью: подготовку текстов и структуры проекта, запись и валидацию аудио, обучение модели, экспорт в ONNX и офлайн-синтез готового голоса.

# Запуск UI если ве настроено:
```bash
python -m app.main record --project vladimir-tts-studio --port 8765
```
> python -m app.main record --project  <Имя вашего проекта> --port 8765

## Windows (Training) — Quick Start

### 0) Что должно быть установлено в системе (один раз)

- ### 0.1 Git Bash - PowerShell
```powershell
git --version
```
- ### 0.2 Python 3.11+ (лучше 3.12) - PowerShell
```powershell
py -v
```
- ###  0.3 espeak-ng (обязательно) - PowerShell
```powershell
espeak-ng --version
```
- ### 0.4 Visual Studio 2022 Build Tools (обязательно для monotonic_align) с CMake tools (часто полезно):
- Открыть: x86_x64 Cross Tools Command Prompt for VS 2022
* Проверка:
```powershell
where cl
```
- Если нет x86_x64 Cross Tools Command Prompt for VS 2022 и Visual Studio 2022 Build Tools , можно установить:
```powershell
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --source winget `
  --accept-source-agreements --accept-package-agreements `
  --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"
```
## Ставим окружение (PowerShell):
- ### 1.1 Создать venv и активировать
- Открой PowerShell в корне репозитория (пример: F:\vladimir-tts-studio)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- Если Windows ругается на политики выполнения:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```
- ### 1.2 Установить зависимости проекта
```powershell
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements\train.txt
python -m pip install -e .
```
- Проверка, что runtime живой:
```powershell
python -c "import importlib; importlib.import_module('piper.espeakbridge'); print('OK espeakbridge')"
```
## 2) Подкачать training-исходники Piper (PowerShell)
- ### Из корня репоизитория:
```powershell
git clone https://github.com/OHF-Voice/piper1-gpl.git third_party/piper1-gpl
```
- ### Проверка что папка на месте:
```powershell
dir third_party\piper1-gpl\src\piper\train\vits
```
## 3) Собрать monotonic_align (VS 2022 Prompt)
- ### Открой x86_x64 Cross Tools Command Prompt for VS 2022
(ты стартуешь в C:\Windows\System32> — это нормально)
- ###  3.1 Перейти в корень репоизитория:
```powershell
cd /d F:\vladimir-tts-studio
```
- ###  3.2 Запустить сборку (важно: python из venv)
```powershell
F:\vladimir-tts-studio\.venv\Scripts\python.exe scripts\10_build_monotonic_align.py
```
>Ожидаемый финал:
> - OK espeakbridge
> - [OK] core*.pyd present ...
> - DONE: monotonic_align is built.
# Можно запускать UI проект:
```bash
python -m app.main record --project sandbox_test --port 8765
```
```bash
python -m app.main record --project vladimir-tts-studio --port 8765
```
- ## Проверка зависимостей
- ### Тест обучения
```bash
python -m app.main doctor --project sandbox_test --auto-fix
python -m app.main train  --project sandbox_test --epochs 1 --check
python -m app.main train  --project sandbox_test --epochs 1
```
```bash
python -m app.main doctor --project vladimir-tts-studio --auto-fix
python -m app.main train  --project vladimir-tts-studio --epochs 1 --check
python -m app.main train  --project vladimir-tts-studio --epochs 1
```

Проект ориентирован на полностью локальный сценарий без обязательных облачных сервисов. Основная идея — дать воспроизводимый и прозрачный пайплайн, в котором все этапы от исходного текста до итоговой модели находятся под контролем пользователя. Внутри репозитория есть CLI-приложение, инструменты подготовки датасета, проверка качества манифеста и аудио, интеграция с обучающим контуром Piper и отдельный веб-интерфейс студии записи.

Архитектурно проект разделён на несколько слоёв. Модуль `app` предоставляет пользовательские workflow-команды (подготовка, запись, обучение, экспорт, проверка окружения), модуль `dataset` отвечает за очистку текста, сегментацию и работу с манифестом, а модуль `training` инкапсулирует запуск обучения, preflight-проверки и экспорт артефактов модели. Веб-часть студии записи находится в `studio/web` и работает как локальный интерфейс для сбора голосовых данных.

Для обучения ожидается стандартная структура проекта с `metadata/train.csv` и WAV-файлами, а также корректное окружение с зависимостями синтеза и тренировки. Валидация перед запуском обучения проверяет импортируемость ключевых модулей, наличие данных, целостность путей в манифесте и базовые аудио-параметры. Это снижает риск типичных ошибок, когда запуск обучения завершается из-за неконсистентного датасета или неподготовленной среды.

Итогом работы `vladimir-tts-studio` является обученная голосовая модель, пригодная для дальнейшего экспорта и использования в inference-сценариях. Репозиторий подходит как для персональных экспериментов с клонированием голоса, так и для регулярной итеративной дообучаемой разработки, где важно последовательно улучшать качество датасета и звучание модели.
