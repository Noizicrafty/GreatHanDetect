# GreatHanDetect — адаптация обнаружения курения для LinQ TPU

**Назначение.** Воспроизводимый инженерный проект адаптации [clyv/smoking-detection](https://github.com/clyv/smoking-detection) для LinQ TPU Framework **2026.9.45** (TPU_H1M).

**Статус:** исходный код интеграции опубликован. Настоящая компиляция для TPU, запуск на LinQ и качество распознавания на CCTV не подтверждены. Веса, ONNX-бинарники, пользовательские датасеты, SDK и готовые `.tpu` в репозиторий не включены.

## Архитектура

1. YOLO-World обнаруживает сигареты, вейпы, сигары, трубки, дым.
2. YOLO11n-Pose определяет позу людей.
3. CPU декодирует сетевые выходы, выполняет NMS и IoU-сопоставление людей между кадрами.
4. Эвристики исходного проекта оценивают движение руки относительно лица и формируют событие предполагаемого курения.
5. Результат: MP4 с разметкой и JSON со списком событий.

Точная совместимость YOLO-World и YOLO11n-Pose с SDK не подтверждена. В официальном перечне проверенных LinQ-моделей они не указаны; указан другой вариант `yolov8n_pose` с нестандартными отсечёнными выходами.

## Windows: исходный CPU/GPU пайплайн

Установите Python 3.12 и Git. В PowerShell:

```powershell
git clone https://github.com/Noizicrafty/GreatHanDetect.git
cd GreatHanDetect
.\setup_windows.bat
.\START_CPU_BASELINE_WINDOWS.bat
```

Для NVIDIA CUDA вместо последнего BAT используйте `START_GPU_BASELINE_WINDOWS.bat`.

Базовый исходный код загружается автоматически в `vendor/smoking-detection` в зафиксированной версии `e7bba89b0336119363cf4d9f9e31b2e4a67e9233`. Это не обученная специально на промышленных записях модель.

## Экспорт ONNX

```powershell
.\EXPORT_AND_INSPECT_WINDOWS.bat
```

Или команды вручную:

```bash
python tools/bootstrap_source.py
python tools/export_onnx.py --part both --device cpu
python tools/inspect_onnx.py --part detector
python tools/inspect_onnx.py --part pose
python tools/run_onnx_reference.py --video test.mp4 --max-frames 100
```

Используются два статических ONNX с batch=1, размером 640, RGB/255, NCHW и letterbox. NMS вынесен на CPU.

## Ubuntu: LinQ TPU

Требования документации пользователя: Ubuntu 18.04 или новее, Python **3.12.11**, пакеты TPU Framework из поставки производителя и устройство TPU.

```bash
bash setup_ubuntu.sh
bash INSTALL_VENDOR_PACKETS_UBUNTU.sh /path/to/packets
source .venv/bin/activate

python tools/bootstrap_source.py
python tools/export_onnx.py --part both --device cpu
python tools/inspect_onnx.py --part detector
python tools/inspect_onnx.py --part pose

python tools/make_calibration.py --part detector --images /data/cctv_frames --limit 64
python tools/make_calibration.py --part pose --images /data/cctv_frames --limit 64

bash COMPILE_BOTH_UBUNTU.sh
```

При успешной реальной компиляции должны появиться два TPU-программных файла (по одному для обнаружения объектов и поз), но это **не гарантируется** — часть ONNX-операторов может быть неподдерживаемой.

Если устройство обнаружено через `pytpu.Device.list_devices()` и получены реальные TPU-файлы:

```bash
python tools/run_tpu_video.py --video /data/test.mp4 --show --max-frames 100
```

Результат: `results/tpu_pipeline`, MP4 и JSON.

## Тесты

```bash
python -m pip install -r requirements/runtime.txt
python -m unittest discover -s tests -v
python tools/doctor.py
```

5 тестов предобработки и декодирования были локально пройдены при сборке. Они не подтверждают точность курения.

## Документы

- [Фактический статус](docs/BUILD_STATUS.md)
- [Совместимость и ограничения](docs/COMPATIBILITY.md)
- [Критерии приёмки](docs/ACCEPTANCE_CHECKLIST.md)

**Важно:** предварительная эвристика определяет возможное курение и может ошибаться при разговорах по телефону, приёме пищи и плохом освещении. Обязательно проверить точность и задержку на размеченных кадрах предприятия. Требуется отдельно проверить лицензии upstream, Ultralytics, датасетов и TPU SDK.
