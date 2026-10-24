# Реестр моделей

Бинарные файлы в Git **не включены**. Запросить их у команды, сохранить в указанные пути и проверить SHA-256. Ссылка на автоматическое скачивание отсутствует. Статический manifest: [model_registry.json](evidence/model_registry.json).

| Роль | Модель | ONNX | TPU | Проверка |
|---|---|---|---|---|
| person | YOLOv8n COCO | `assets/onnx/yolo8_person.onnx` | `assets/tpu/yolo8_person.tpu` | Полный сценарий зоны |
| pose | YOLO11n-Pose | `assets/onnx/person_pose.onnx` | `assets/tpu/person_pose.tpu` | Реальное исполнение, без keypoint AP |
| detector | YOLO-World, cigarette/vape/cigar/pipe/smoke | `assets/onnx/smoking_objects.onnx` | `assets/tpu/smoking_objects.tpu` | Исполняется; качество PTQ не принято |
| trained detector | YOLOv8n, cigarette/person/smoke/vape | Отдельный `last.pt` | Не скомпилирован | Локальные предсказания после 26 эпох |

Первые три модели: вход `images float32 [1,3,416,416]`, выход `output0 float32`, каналы/позиции соответственно `[1,84,3549]`, `[1,56,3549]`, `[1,9,3549]`. RGB/255, NCHW, letterbox, CPU NMS. Обученный `.pt` не взаимозаменяем с пяти-классовым YOLO-World: нельзя просто заменить путь без согласования классов/выходов.

## Хеши проверенных файлов

### person

- ONNX SHA-256: `b0c7cdeda052d2a1a7598660c8a7e754320310f23c117436f0631cab1d87e6db`
- TPU SHA-256: `5d3a9bf38c04380b6acc3396f828779bd181f50d92b20abefe4044d18007e840`

### detector

- ONNX SHA-256: `57faaf1cbde3f4df74ae15e1ca16329b3b8b3f37376b753390fa770a6fc43ea7`
- TPU SHA-256: `7d47261f7d0a6911f2444f483c2a4974c48c3b6f7512086f11da2de2b26b2fbc`

### pose

- ONNX SHA-256: `9bbc952eeb6e408d198654a79cac678c79cfe23ac28b89a621d440e743203c87`
- TPU SHA-256: `4544e395d34743788df0f806f5bf81f3ad85e9fabc72b24808db50ef4aab3acd`

### Trained checkpoint

- Last: 26 эпох, SHA-256 `5b9df5d26785a3848643b7603036c2219dea4aedd6a19373dadaaa56182e519d`.
- Best: 22 эпох (внутреннее epoch=21); выбран по валидации, не test.
- Веса сохраняются отдельно у команды. Нет итогового test AP, интеграции или проверки этих весов на TPU.

Проверка файла: PowerShell `Get-FileHash -Algorithm SHA256 <path>`; Linux `sha256sum <path>`.
