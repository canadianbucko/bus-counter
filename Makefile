.PHONY: run docker-up docker-down deps detect-test

run:
	go run ./cmd/app

deps:
	python3 -m pip install -r detector/requirements.txt

docker-up:
	docker compose up --build

docker-down:
	docker compose down

detect-test:
	python3 detector/detect.py --input test.jpg --output data/outputs/test_result.jpg --model yolov8n.pt --target bus --conf 0.25
