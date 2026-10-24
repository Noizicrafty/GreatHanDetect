# QA локального демо — 24.10.2026

Статус: **все перечисленные проверки закрыты, открытых дефектов в проверенном наборе нет**. Это не заявление об отсутствии любых возможных ошибок или о промышленной готовности.

Демо: http://127.0.0.1:8090/. Проверено в Microsoft Edge на Windows. Этот раздел фиксирует историческую локальную проверку 24.10.2026. На 25.10.2026 демо опубликовано в GitHub; актуальные результаты Compose и браузера — [reviewer_checks.json](evidence/reviewer_checks.json), запуск — [REVIEWER_RU.md](REVIEWER_RU.md).

## Итог

- **26** интеграционных серверных тестов — PASS.
- **22** браузерные проверки, включая все пять страниц — PASS.
- **59** регрессионных тестов основного приложения — PASS.
- Реальный CPU-инференс checkpoint после **26 эпох** проверен; обучение не запускалось.
- Проверен именно PDF, скачанный кнопкой браузера: бинарный PDF, корректное имя, кириллица, разбор содержимого и визуальный рендер.
- Проверена пагинация 105 событий: **7 страниц**, повтор заголовков и отсутствие текста за границами страниц.

## Закрытые дефекты

| Дефект | Исправление | Проверка |
|---|---|---|
| Скачивание исходного статичного отчёта вместо текущих данных | Кнопка и iframe сразу в HTML обращаются к одному динамическому PDF API | Browser PDF download + API preview/download |
| Тесная строка числовых показателей PDF | Фиксированная высота строк, leading и вертикальное выравнивание, встроенный кириллический шрифт | Рендер скачанного PDF |
| Сумма решений не объясняла общее число событий | Отдельная колонка «Авто»; автоматические записи не выданы за решения оператора | Сверка журнала и PDF |
| Кнопки решений/Excel в исходном макете без действий | API, SQLite, настоящая XLSX-выгрузка с фильтрами | Server + browser |
| Ошибка загрузки изображения в BytesIO | Явный формат JPEG и валидация изображения до декодирования | Настоящий checkpoint + ошибочные файлы |
| Незакрытые соединения SQLite | Явные commit/rollback/close через context manager | Перезапуск, повторные запросы, закрытие временной БД |
| Повторная загрузка кадра создавала дубли | Ключ по нормализованному изображению и весам; решение сохраняется | Повторный predict после подтверждения |
| Архивные GIF были однокадровыми | Анимации камер взяты из 29-кадрового samples.gif; бег показан как статичный пример | Декодирование GIF + browser play/pause |
| На узком экране отсутствовала навигация | Добавлена нижняя мобильная навигация | Edge 390 и 320 px |
| Пустые результаты поиска без пояснения | Добавлено сообщение о пустом результате | Фильтры/поиск |
| favicon давал 404 в консоли | Корректный пустой ответ 204 | API + browser |

## Серверная матрица

| Тест | Статус |
|---|---|
| `test_01_root_and_five_routes` | PASS |
| `test_02_all_referenced_assets_exist` | PASS |
| `test_03_state_is_explicit_demo_and_five_events` | PASS |
| `test_04_confirm_is_persistent_and_idempotent` | PASS |
| `test_05_reject_updates_journal` | PASS |
| `test_06_unknown_and_invalid_review_do_not_mutate` | PASS |
| `test_07_reset_restores_fixtures` | PASS |
| `test_08_invalid_json` | PASS |
| `test_09_bad_origin_and_host` | PASS |
| `test_10_traversal_and_unknown_routes` | PASS |
| `test_11_oversized_request` | PASS |
| `test_12_pdf_download_headers_and_binary` | PASS |
| `test_13_pdf_preview_matches_download_content` | PASS |
| `test_14_pdf_cyrillic_and_counts_follow_review` | PASS |
| `test_15_pdf_multiple_pages_and_no_out_of_bounds_text` | PASS |
| `test_16_xlsx_opens_and_matches_journal` | PASS |
| `test_17_xlsx_category_and_search_filters` | PASS |
| `test_18_dataset_excludes_demo_events` | PASS |
| `test_19_gif_animation_assets` | PASS |
| `test_20_favicon_is_not_an_error` | PASS |
| `test_21_positive_model_duplicate_review_and_pseudo_labels` | PASS |
| `test_22_negative_frame_and_busy_response` | PASS |
| `test_23_bad_base64_is_rejected` | PASS |
| `test_24_invalid_json_shape_and_non_image` | PASS |
| `test_25_parallel_pdf_exports` | PASS |
| `test_26_empty_journal_exports` | PASS |

## Браузерная матрица

| Проверка | Статус |
|---|---|
| UI-01 initial pending notifications restored | PASS |
| UI route and active navigation: cameras | PASS |
| UI route and active navigation: monitoring | PASS |
| UI route and active navigation: analytics | PASS |
| UI route and active navigation: notifications | PASS |
| UI route and active navigation: report | PASS |
| UI-02 camera animation starts | PASS |
| UI-03 camera animation pauses to still | PASS |
| UI-04 confirm survives reload | PASS |
| UI-05 reject survives reload | PASS |
| UI-06 empty notifications state | PASS |
| UI-07 category filter | PASS |
| UI-08 empty search | PASS |
| UI-09 Excel download filename | PASS |
| UI-10 table sort | PASS |
| UI-11 actual 26 epoch inference in browser | PASS |
| UI-12 reviewed dataset download | PASS |
| UI-13 PDF download filename | PASS |
| UI-14 mobile layout without page overflow | PASS |
| UI-16 mobile navigation works | PASS |
| UI-17 narrow 320px viewport without page overflow | PASS |
| UI-15 no uncaught JavaScript errors | PASS |

## Что проверено дополнительно

Скачанный XLSX открыт через независимый reader: фильтр «каска» даёт две строки, формат confidence сохранён. ZIP прочитан, manifest указывает псевдометки, изображения и YOLO labels присутствуют, DEMO-кадры исключены. Состояние решений сохраняется после повторного открытия SQLite и обновления браузера. Положительный и отрицательный кадры проверены настоящими весами; конкурентный инференс возвращает понятный busy-ответ.

## Ограничения

Это локальный показ интерфейса, пользовательских записей и покадровой модели. PPE/страховочный трос/бег из презентации — DEMO. Один кадр с сигаретой не является доказанным временным эпизодом. Реальная LinQ-интеграция основного приложения существует отдельно и в этой демке не вызывается. Заявления о 98%/78% точности и <1 с не подтверждены этими тестами.

## Артефакты

Исходный набор проверок — `tools/test_presentation_demo.py`; серверные результаты — `results/presentation_demo_qa/backend_qa.json`; логи базовых тестов — `results/core-regression-tests.log`. Браузерные скриншоты, скачанные файлы и manifest находятся в папке `outputs/demo_qa` рабочего пространства. Тестовый большой журнал создан отдельно; боевой демонстрационный журнал им не загрязнён.
