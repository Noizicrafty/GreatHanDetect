# Сторонние материалы и атрибуция

- **clyv/smoking-detection**: https://github.com/clyv/smoking-detection, pinned commit `e7bba89b0336119363cf4d9f9e31b2e4a67e9233`. Загружается через `tools/bootstrap_source.py`, не включается копией в Git. Отдельный LICENSE в проверенной версии не найден; не заявлять MIT или разрешение на произвольное переиздание. Условия требуют уточнения перед распространением продукта.
- **Ultralytics YOLO и веса**: https://github.com/ultralytics/ultralytics. У проекта отдельные AGPL/коммерческие условия; не считать собственную модификацию разрешением на релицензирование зависимостей или весов.
- **Richie Lab / smoking-tasfx v4**: https://universe.roboflow.com/richie-lab/smoking-tasfx/dataset/4. CC BY 4.0 согласно метаданным предоставленного экспорта: https://creativecommons.org/licenses/by/4.0/. `docs/images/` — производные иллюстрации исходных фотографий с рамками модели и компоновкой; данная ссылка служит атрибуцией источника, не идентифицирует изображённых людей. Происхождение и хеши кадров указаны в `docs/evidence/trained_predictions.json`. Публикуются небольшие иллюстрации и один готовый пример проверки в demo/ff/assets/photos/trained_sample.jpg, не весь датасет.
- **Публичные bus.jpg / zidane.jpg** использовались для инженерного сравнения стандартной модели; не выдаются за собственные производственные данные. В основных иллюстрациях checkpoint используется указанный Roboflow-экспорт.
- **PaddleDetection**: https://github.com/PaddlePaddle/PaddleDetection. Ссылка для возможных расширений. Его код, модели и их исполнение не включены в текущую DEMO-связку.
- **LinQ TPU Framework 2026.9.45 и codec wheel**: поставка организаторов/производителя с собственными условиями. SDK, пакеты, PDF и VPN-файлы в Git не распространяются.

Прототип не объявляет единую лицензию на все перечисленные материалы. До внешнего коммерческого развёртывания требуется закончить проверку условий их использования.

## MediaPipe webcam components

Webcam mode bundles @mediapipe/tasks-vision 0.10.21 and the official Face Landmarker float16/1 asset. Bundled Apache-2.0 license and upstream URLs/file hashes: demo/ff/assets/mediapipe/LICENSE and manifest.json. Original npm vision_bundle.cjs is served as vision_bundle.js without code changes. The face model is separate from the trained cigarette checkpoint; this mode does not establish event-level accuracy.

## Материалы презентационного демо

`demo/ff` содержит предоставленные командой макеты, фотографии и анимации из final_frames.rar и samples.gif. Публикуются по запросу команды для проверки хакатон-демо; не объявляются собственными съёмками или свободно лицензированным датасетом. Веса после 26 эпох и TPU-программы публикуются в отдельном Release; это не меняет условий Ultralytics и исходных данных.
