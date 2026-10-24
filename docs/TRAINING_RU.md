# Данные, обучение и перенос на LinQ

**Обучение необязательно и сейчас остановлено по запросу команды.** Последний checkpoint — после 26 эпох, best — после 22. Для установки и демонстрации обучение запускать не нужно. Команды ниже предназначены для будущего осознанного запуска.

В прерванном прогоне не выполнены финальная оценка на test и автоматический экспорт ONNX. Перечисленные выходные артефакты создаются при штатном завершении, их наличие не заявляется. [Реальные предсказания сохранённых весов](TRAINED_MODEL_RESULTS_RU.md).

Источник: [Richie Lab, smoking-tasfx v4](https://universe.roboflow.com/richie-lab/smoking-tasfx/dataset/4), CC BY 4.0. Полный экспорт предоставлен командой. Исходные файлы остаются неизменными.

## Подготовка

```bash
python tools/audit_dataset.py --root "/path/to/train data" --out results/dataset_audit
python tools/curate_smoking.py --audit results/dataset_audit --out assets/datasets/smoking_v4
```

Аудит декодирует изображения, проверяет YOLO-разметку, считает классы и ищет точные/близкие совпадения между выборками. Сохраняется тестовая часть; связанные с ней примеры удаляются из валидации и обучения. После этого из обучения удаляются совпадения с оставшейся валидацией. Порог perceptual hash — 4 бита, учитывается горизонтальное отражение. Это не гарантирует независимость актёров/видеосюжетов.

В данной выгрузке: 12 046 исходных обучающих, 318 валидационных и 122 тестовых изображения, без ошибок формата. После проверки: **11 897 / 309 / 122**, исключено 158 межвыборочных совпадений. Эти фактические числа отличаются от счётчиков опубликованной версии; использовать результаты локального аудита.

`Cigarette` и `smoking` объединяются в `cigarette` после визуальной проверки обучающих аннотаций. Остальные классы — `person`, `smoke`, `vape`. Производная копия использует новые файлы разметки и hardlink изображений, при невозможности hardlink — копии. Исходные подписи и разметка не редактируются. Таблица преобразования сохраняется в `curation_manifest.json`.

## GPU-обучение

Создайте отдельное окружение. Для проверенной Windows GPU-среды использованы PyTorch 2.9.1+cu128 / torchvision 0.24.1 из официального индекса PyTorch, Ultralytics 8.4.186 и RTX 4070 SUPER 12 GB. Клиенту приложения CUDA не нужна.

```bash
python -m venv .venv-train
# Активировать .venv-train в своей оболочке.
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/train.txt
python tools/train_smoking.py --data assets/datasets/smoking_v4/data.yaml --out results/training_roboflow_v4 --epochs 40 --imgsz 640 --batch 16 --workers 4 --device 0 --patience 8
```

Seed — 42, AMP и deterministic включены. Редкие обучающие изображения с `vape` повторяются четыре раза; валидация и тест остаются без повторения. Сохраняются manifest, хеши списков выборок, CSV обучения, checkpoints, тестовые метрики и ONNX для размеров 416 и 640. Выбор лучшего checkpoint выполняется по валидации; тест не используется для ранней остановки.

`training_status.json` служит удобным индикатором стадии, но перед повторным запуском проверяйте реальный процесс, а не только этот файл. Проверка на тесте — объектная оценка изображений, не оценка эпизодов курения. В тесте этой выгрузки нет объектов `vape`, поэтому полноту этого класса по нему определить нельзя.

## Калибровка и компиляция

Для калибровки выбирайте изображения **из отфильтрованной обучающей части**, включая разные масштабы людей, сигареты, дым и трудные отрицательные примеры. Не берите тестовые изображения и не используйте случайный шум как доказательство качества.

```bash
python tools/calibrate_application.py --config /path/to/model.application.json --model detector --images /path/to/train1.jpg /path/to/train2.jpg --out assets/calibration/smoking.npz
```

В отдельной Linux/Python 3.12.11 среде с SDK:

```bash
bash INSTALL_VENDOR_PACKETS_UBUNTU.sh /path/to/vendor/packets
python tools/compile_application.py --config /path/to/model.application.json --model detector --calibration assets/calibration/smoking.npz --out assets/tpu/smoking.tpu
```

Компиляция использует `preliminary_model_conditioning`, `RegularModel.calibrate_torch`, quantize, `Network.from_quantized_model` и `compiler.compile_` с `TPU_H1M / O1`. Успешная компиляция сама по себе не подтверждает качество. После неё нужны сравнение выходов с CPU и оценка на тех же отложенных изображениях.

## Проверка событий

Формат эталона/предсказаний: JSON-массив объектов с `camera_id`, `zone_id`, `event_type`, `start_s`, `end_s`. Затем:

```bash
python -m greathan.evaluation --predictions predictions.json --truth truth.json --duration-s 600 --minimum-iou 0.1 --out results/event_quality.json
```

Сопоставление — один к одному; лишние события учитываются как FP, дубли считаются отдельно. Указывать фактические минуты наблюдения, число независимых роликов и условия совпадения. Короткие или повторяемые сцены не доказывают низкую частоту ложных тревог за рабочую смену.
