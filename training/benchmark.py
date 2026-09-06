"""Train/benchmark RoadLens on the public Nepal plate dataset.

This script never invents metrics. It writes measured metrics only after a real
training/validation run. The test split is evaluated separately and its metrics
are stored with the dataset and model metadata.

Examples:
  python training/benchmark.py --device 0 --epochs 50
  python training/benchmark.py --weights training/runs/nepal_plate_detector/weights/best.pt --device 0
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from train_nepal_detector import prepare, DATASET


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="Existing trained weights; if omitted, train first")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    data = prepare()
    weights = args.weights
    if not weights:
        model = YOLO(args.model)
        model.train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, patience=15, cache=False, workers=2,
            degrees=8, translate=0.08, scale=0.5, shear=3, perspective=0.0005,
            fliplr=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.5,
            project="training/runs", name="nepal_plate_detector", exist_ok=True,
        )
        weights = "training/runs/nepal_plate_detector/weights/best.pt"

    model = YOLO(weights)
    # Explicitly evaluate the untouched test split. Ultralytics documents that
    # split="test" is required for a held-out test measurement.
    result = model.val(data=str(data), split="test", imgsz=args.imgsz,
                       batch=args.batch, device=args.device, plots=True)

    metrics = {
        "status": "measured",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "split": "test",
        "weights": str(weights),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "metrics": {
            "precision": float(result.box.mp),
            "recall": float(result.box.mr),
            "mAP50": float(result.box.map50),
            "mAP50_95": float(result.box.map),
        },
    }
    out = Path("training/runs/nepal_plate_detector/test_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote measured test metrics to {out}")


if __name__ == "__main__":
    main()
