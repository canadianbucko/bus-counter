# Подсчет автобусов на остановке

## Запуск

```bash
docker compose up --build
```


```text
http://localhost:8080
```

## Без докера

```bash
python3 -m pip install -r detector/requirements.txt
go run ./cmd/app
```

## API

```text
POST /api/v1/detect
GET /api/v1/history
DELETE /api/v1/history
GET /api/v1/reports/csv
```
