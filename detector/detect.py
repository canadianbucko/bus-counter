#!/usr/bin/env python3

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--target", default="bus")
    parser.add_argument("--conf", default="0.25")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    try:
        with redirect_stdout(sys.stderr):
            from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("нет ultralytics") from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError("файл не найден")

    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError("не удалось открыть картинку")

    with redirect_stdout(sys.stderr):
        model = YOLO(args.model)
        results = model.predict(source=image, conf=float(args.conf), verbose=False)
    result = results[0]

    target = args.target.lower().strip()
    names = result.names or {}
    objects = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = str(names.get(cls_id, cls_id)).lower()
            if class_name != target:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())

            objects.append({
                "class": class_name,
                "confidence": round(conf, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            })

    annotated = image.copy()
    for obj in objects:
        x1, y1, x2, y2 = map(int, [obj["x1"], obj["y1"], obj["x2"], obj["y2"]])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
        label = f'{obj["class"]} {obj["confidence"]:.2f}'
        cv2.putText(annotated, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

    cv2.imwrite(str(output_path), annotated)

    print(json.dumps({
        "count": len(objects),
        "objects": objects,
        "elapsed_ms": int((time.time() - start) * 1000),
        "model": args.model,
        "target": target,
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
