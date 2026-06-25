FROM golang:1.22-bookworm AS builder
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/bus-counter ./cmd/app

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HTTP_ADDR=:8080 \
    DATA_DIR=/app/data \
    DETECTOR_PATH=/app/detector/detect.py \
    PYTHON_BIN=python3 \
    MODEL_PATH=yolov8n.pt \
    TARGET_CLASS=bus \
    CONFIDENCE=0.25

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY detector/requirements.txt /app/detector/requirements.txt
RUN pip install --no-cache-dir -r /app/detector/requirements.txt

COPY --from=builder /out/bus-counter /app/bus-counter
COPY detector /app/detector
COPY web /app/web
COPY data /app/data

EXPOSE 8080
CMD ["/app/bus-counter"]
