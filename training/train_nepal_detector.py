"""Fine-tune a Nepal plate detector from public data.

Designed for a free Colab/Kaggle CPU/GPU session or a local machine. It downloads
only openly accessible datasets and never uploads CCTV frames.

Usage:
  python training/train_nepal_detector.py --epochs 50 --device 0

The script uses the CC-BY-4.0 Nepali Private License Plates dataset. The dataset
contains 1,172 source photos and four augmented variants per source (5,860
image/label pairs). We split by source-family to avoid augmentation leakage.
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from ultralytics import YOLO

DATASET = "mukulboro/nepali-private-license-plates"
ROOT = Path("training/data/nepali_private")


def source_family(stem: str) -> str:
    for suffix in ("_blur", "_contrast", "_exposure", "_noise"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def stable_split(key: str) -> str:
    n = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    return "val" if n < 15 else "test" if n < 25 else "train"


def prepare():
    raw = Path(snapshot_download(repo_id=DATASET, repo_type="dataset", local_dir=ROOT / "raw"))
    image_dir = ROOT / "images"
    label_dir = ROOT / "labels"
    for split in ("train", "val", "test"):
        (image_dir / split).mkdir(parents=True, exist_ok=True)
        (label_dir / split).mkdir(parents=True, exist_ok=True)

    # The HF dataset layout can change. Discover images and matching YOLO labels.
    images = [p for p in raw.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    labels = {p.stem: p for p in raw.rglob("*.txt") if p.name != "labels.txt"}
    copied = 0
    for img in images:
        label = labels.get(img.stem)
        if label is None:
            continue
        split = stable_split(source_family(img.stem))
        shutil.copy2(img, image_dir / split / img.name)
        shutil.copy2(label, label_dir / split / f"{img.stem}.txt")
        copied += 1

    if copied < 100:
        raise RuntimeError(f"Only found {copied} image/label pairs. Inspect {raw} and update the dataset adapter.")

    yaml = ROOT / "data.yaml"
    yaml.write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: embossed\n  1: provincial\n  2: regional\n",
        encoding="utf-8",
    )
    print(f"Prepared {copied} labeled images: {yaml}")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 for GPU, cpu for CPU")
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()
    data = prepare()
    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=15, cache=False, workers=2,
        degrees=8, translate=0.08, scale=0.5, shear=3, perspective=0.0005,
        fliplr=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.5,
        project="training/runs", name="nepal_plate_detector", exist_ok=True,
    )
    metrics = model.val(data=str(data), imgsz=args.imgsz, device=args.device)
    print("Validation metrics:", metrics)
    print("Best weights: training/runs/nepal_plate_detector/weights/best.pt")


if __name__ == "__main__":
    main()
