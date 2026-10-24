# Совместимость clyv с LinQ TPU

| Компонент | В исходной программе | На TPU |
|---|---|---|
| Object detector | YOLO-World yolov8l-worldv2.pt | Не в проверенном списке |
| Pose | yolo11n-pose.pt | Не в проверенном списке, есть другая yolov8n_pose |
| Tracking | ByteTrack | В адаптере CPU IoU tracker |
| Курение | CPU heuristics / optional temporal classifier | CPU эвристика |
| Вход | BGR frame | RGB / 255, NCHW, letterbox |
| Выход | Ultralytics Results | CPU декодирование raw heads + NMS |
| Runtime | CUDA/CPU | pytpu и устройство /dev/tpu* |

Риски: неподдерживаемые ONNX-операторы, нестандартные обрезанные головы YOLO, ошибки ввода/вывода, задержка, ошибки PTQ и лицензии.
Только успешная реальная компиляция SDK создаёт исполняемые `.tpu`, а не переименование `.onnx`.
Тестирование на RTX 4060 не заменяет измерение на физическом LinQ TPU.

Источник спецификации: предоставленный пользователем tpu_framework.pdf, выпуск 2026.9.45, главы 5–6.
