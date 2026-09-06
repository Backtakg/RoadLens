# RoadLens — Nepal ANPR

RoadLens is a **zero-paid-API-cost, self-hosted** Automatic Number Plate Recognition system for Nepal plates and authorized CCTV/RTSP feeds.

## Accuracy-first architecture

```text
Authorized CCTV / webcam
        ↓
YOLO Nepal plate detector(s) + ByteTrack
        ↓
plate crop + illumination/weather preprocessing
        ↓
Tesseract (Nepali + English) ─┐
PaddleOCR (PP-OCRv5, Nepali) ─┼→ OCR ensemble
optional second detector ──────┘
        ↓
Nepal-aware normalization + temporal consensus
        ↓
SQLite event + dashboard
```

RoadLens is deliberately **not dependent on one detector or one OCR engine**. Ultralytics documents ByteTrack as an available tracker for video streams, and PaddleOCR officially lists Nepali (`ne`) among its PP-OCRv5 supported languages. Those are implementation capabilities, not RoadLens accuracy guarantees.

## Public Nepal data and training

The main detector training source is the public **Nepali Private License Plates** dataset:

https://huggingface.co/datasets/mukulboro/nepali-private-license-plates

Its dataset card states 1,172 unique source photos and 5,860 image/label pairs, with original/blur/contrast/exposure/noise variants, under CC BY 4.0. RoadLens's training script groups the variants belonging to the same original photo before splitting train/validation/test to avoid augmentation leakage.

### Train your Nepal detector

Use a free GPU notebook such as Colab/Kaggle or your own machine:

```bash
pip install -r requirements.txt
python training/train_nepal_detector.py --epochs 50 --device 0
```

Then run the resulting weights:

```bash
PLATE_MODEL_PATH=training/runs/nepal_plate_detector/weights/best.pt python app.py
```

A second independently trained detector can be supplied with:

```bash
PLATE_MODEL_PATH_2=/path/to/second/best.pt
```

RoadLens merges overlapping detections from the detector ensemble.

## Existing external models

RoadLens can also use this publicly available Nepal plate detector by default:

https://huggingface.co/krishnamishra8848/Nepal-Vehicle-License-Plate-Detection

That model card reports P=0.973, R=0.956, mAP@50=0.988 and mAP@50-95=0.929 **for the publisher's own evaluation**. These numbers are not RoadLens measurements and are never presented as RoadLens accuracy.

## No fabricated accuracy numbers

**Current RoadLens deployment accuracy: not yet benchmarked.**

The repository does not claim that RoadLens is 99%, 95%, or any other percentage accurate. A public model's score is not a substitute for testing RoadLens on held-out data from the intended cameras.

The evaluation tools measure:

- plate exact-match accuracy
- character error rate
- missed-read rate
- detection precision/recall/mAP when detector labels are available
- false reads per 1,000 vehicles
- day/night/rain/fog/glare breakdowns
- track-level exact-match accuracy

See [`evaluation/README.md`](evaluation/README.md) and [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

### Evaluation

Create a labeled CSV containing real held-out predictions:

```csv
image_id,ground_truth,prediction
```

Then run:

```bash
python evaluation/plate_metrics.py results.csv --out evaluation/metrics.json
```

The repository's example rows, if any, are **format examples only**, not benchmark data.

## Weather and difficult conditions

RoadLens includes multiple local preprocessing variants: CLAHE, denoising, sharpening, adaptive thresholding and gamma variants, followed by multi-frame consensus. These techniques are intended to improve robustness; they do **not** prove all-weather accuracy.

For real deployment, collect an authorized, labeled, held-out set from the exact cameras covering daylight, night, rain, fog, glare, motion blur, vehicle distance, perspective and plate types. Freeze the test set before tuning thresholds.

## Run with Docker

```bash
docker compose up --build
```

Open `http://localhost:5000`.

For a local USB camera, use `0` (or another camera index).

For an authorized CCTV camera:

```text
rtsp://user:password@192.168.1.50:554/stream
```

The RoadLens machine opens the RTSP stream and sends an annotated MJPEG stream to the browser.

## Run without Docker

```bash
pip install -r requirements.txt
python app.py
```

For the optional local PaddleOCR backend, install a compatible PaddlePaddle build and:

```bash
pip install -r requirements-accuracy.txt
```

## Environment variables

```text
PLATE_MODEL_PATH=/path/to/best.pt     # your trained detector
PLATE_MODEL_PATH_2=/path/to/other.pt  # optional ensemble detector
PLATE_CONF=0.28
PROCESS_EVERY_N=2
ROADLENS_PADDLE=1
OCR_LANG=nep+eng
```

## Accuracy-critical camera setup

Model quality is only half the problem. Aim the camera so plates occupy enough pixels, minimize extreme perspective, use an appropriate shutter/IR setup at night, and avoid direct headlight saturation. A high-resolution camera poorly aimed at a distant road can perform worse than a lower-resolution camera positioned correctly.

## Privacy / authorization

Only connect CCTV streams you are authorized to access and process plate data under an appropriate legal basis. Do not use RoadLens to access third-party camera systems without permission.
