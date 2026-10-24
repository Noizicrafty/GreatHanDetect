# Развёртывание LinQ Demo на хосте

Дата проверки: 24.10.2026. Публичный адрес: **https://linq.user-link.xyz/**.

## Состав

- Сервер: `27.123.245.104`, Debian 13, Docker 29.3.0 / Compose 5.1.0.
- Рабочая папка: `/opt/linq-demo`.
- Compose: `/opt/linq-demo/compose.linq-demo.yaml`.
- Контейнер: `linq-safety-demo`, образ `greathan-linq-demo:2026-10-24`.
- Внутренний порт приложения 8090 опубликован только на `127.0.0.1:18090` хоста. HTTPS обслуживает существующий Nginx.
- Контейнер работает от `10001:10001`, с `restart: unless-stopped`, healthcheck, лимитами 4 CPU / 3 GB и ограничением capabilities.
- Журнал, кадры и результаты сохраняются в named volume `linq_demo_data`. Сохранение ID, состояний и источников событий после перезапуска контейнера проверено.
- Python в образе: 3.12.15; inference PyTorch CPU, сохранённые веса после 26 эпох. Обучение не запускается.

## Домен и сертификат

DNS A домена указывает на сервер. HTTP перенаправляется на HTTPS; корень HTTPS открывает страницу трансляций. Сертификат Let’s Encrypt (YE1), действителен до **22.01.2027**. Внешняя проверка TLS 1.3 выполнена с проверкой цепочки доверия.

- Конфигурация Nginx: `/etc/nginx/sites-available/linq-demo.conf`, ссылка из `sites-enabled`.
- Proxy snippet: `/etc/nginx/snippets/linq-demo-proxy.conf`.
- ACME webroot: `/var/www/linq-acme`.
- Сертификат: `/etc/letsencrypt/live/linq.user-link.xyz/fullchain.pem`.
- Автопродление: `snap.certbot.renew.timer`, включён. Успешный `renew --dry-run` для этого домена подтверждён.
- Deploy hook сертификата выполняет `systemctl reload nginx`.

## Управление

```bash
cd /opt/linq-demo
docker compose -f compose.linq-demo.yaml ps
docker compose -f compose.linq-demo.yaml logs --tail 100 demo
docker compose -f compose.linq-demo.yaml restart demo
```

Обновление исторического стенда (веса теперь загружаются из фиксированного Release с SHA-256):

```bash
docker compose -f compose.linq-demo.yaml build --pull
docker compose -f compose.linq-demo.yaml up -d --no-build
nginx -t
```

Данные находятся в отдельном томе и сохраняются при обычном пересоздании контейнера. Шаблоны Docker/Nginx находятся в `deploy/linq-demo/`. Пароль SSH и другие учётные данные в конфигурацию и пакет поставки не включены.

## Проверки

**27** локальных интеграционных проверок приложения и HTTPS Origin/Host — PASS. **22** браузерные проверки выполнены на публичном домене — PASS: пять разделов, анимации, подтверждение/отклонение и сохранение решений, фильтры/поиск/сортировка, Excel, PDF, ZIP, реальный inference, мобильная навигация на 390/320 px и отсутствие необработанных ошибок JavaScript.

Скачанный с домена Linux-PDF отдельно открыт и отрендерен: кириллица и границы текста корректны. XLSX содержит две строки при фильтре каски. ZIP содержит подтверждённый кадр настоящего checkpoint, псевдометки обозначены явно. SHA-256 модели в результате реального серверного inference совпадает с переданным checkpoint.

Контейнер после restart вернулся в `healthy`, данные сохранились. Проверка публикации порта подтвердила loopback. Certbot staging renewal прошёл успешно. Операционный receipt на сервере: `/opt/linq-demo/deployment-verification.json`; описание релиза: `release-manifest.json`.

## Границы демонстрации

Открытый сайт показывает демонстрационные материалы презентации и реальную покадровую детекцию обученным checkpoint. PPE/страховочный трос/бег помечены DEMO. Обнаружение сигареты в одном кадре не подтверждает временной эпизод курения. LinQ в этой веб-демонстрации не вызывается; отдельная интеграция исходного приложения описана в основной документации. Заявления о 98%/78% точности не выдаются за результат этих испытаний.

## Обновление 25.10.2026: LinQ TPU

Опубликован маршрут /api/demo/webcam/linq через ограниченный reverse SSH relay к JPEG-сервису в командном Jupyter. /api/demo/linq-health возвращает backend и хеш TPU-программы. В веб-камере есть явный выбор CPU/LinQ и пределы 2/5/10 FPS; TPU допускает два кадра в полёте. Проверенный браузерный прогон дал 4,44 кадра/с на виртуальном входе. Подробности и ограничения: [TPU_RESULTS_RU.md](TPU_RESULTS_RU.md).

## Переносимая поставка 25.10.2026

Для нового хоста используйте основной [compose.yaml](../compose.yaml) и [инструкцию кейс-холдеру](REVIEWER_RU.md). `compose.linq-demo.yaml` сохраняется для существующего стенда с loopback 18090 и прежним томом. Он не является основным путём проверки нового клона.
