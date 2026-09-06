# RoadLens — Nepal ANPR

RoadLens is a **zero-paid-API-cost, self-hosted** Automatic Number Plate Recognition system for Nepal plates and authorized CCTV/RTSP feeds.

## The upgraded pipeline

```text
Authorized CCTV / webcam
        ↓
YOLO Nepal plate detector(s) + ByteTrack
        ↓
plate crop + illumination/weather enhancement
        ↓
Tesseract (Nepali + English) ─┐
PaddleOCR (local PP-OCR) ─────┼→ weighted OCR ensemble
optional second detector ─────┘
        ↓
Nepal-aware normalization + temporal consensus
        ↓
SQLite event + dashboard
```

The application is intentionally **not dependent on one OCR engine or one detector**. PaddleOCR and the second detector are optional at Python level but are enabled by the Docker image. ByteTrack keeps the same vehicle/plate track across consecutive frames so RoadLens can vote over several observations instead of trusting one blurry frame.

Ultralytics documents persistent tracking with `persist=True` and ByteTrack/BoT-SORT for continuous streams. citeturn3search0turn3search2

## Public Nepal data and training

The main detector training source is the public **Nepali Private License Plates** dataset. It is CC BY 4.0 and contains 1,172 original photographs plus blur/contrast/exposure/noise variants for 5,860 image/label pairs across three Nepal plate types. citeturn1search0

RoadLens includes a training script that downloads that dataset and **splits by original-image family** so augmented copies do not leak between train/validation/test.

There is also recent Nepali number-plate character research reporting a 34-character dataset with 26,537 labeled character samples and a YOLO + CNN recognition pipeline reaching up to 93% character recognition accuracy. citeturn4search0turn4academia18

### Train your Nepal detector for free

Use a free GPU notebook such as Colab/Kaggle or your own computer:

```bash
pip install -r requirements.txt
python training/train_nepal_detector.py --epochs 50 --device 0
```

Then run the trained model:

```bash
PLATE_MODEL_PATH=training/runs/nepal_plate_detector/weights/best.pt python app.py
```

You can train a second detector and set:

```bash
PLATE_MODEL_PATH_2=/path/to/second/best.pt
```

RoadLens ensembles overlapping detections from both models.

## Why I am not claiming 100% accuracy

No camera-based ANPR system can honestly guarantee perfect recognition in *all* weather. Rain droplets, fog, glare, night exposure, motion blur, compression, oblique plates and too few pixels on the plate can destroy information that no model can recover.

Instead, RoadLens is designed to **measure and improve** accuracy:

- detection precision/recall and mAP
- plate exact-match accuracy
- character error rate
- false reads per 1,000 vehicles
- missed-plate rate
- day/night/rain/fog breakdowns
- track-level exact-match accuracy

For final deployment, fine-tune using authorized footage from the exact cameras, then keep a completely unseen test set.

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

PaddlePaddle currently documents CPU installation for Python 3.9–3.13 and provides CPU/GPU packages; RoadLens uses the local CPU package in its Docker image, so no cloud OCR API is required. citeturn5search0turn5search5

## Environment variables

```text
PLATE_MODEL_PATH=/path/to/best.pt     # your trained detector
PLATE_MODEL_PATH_2=/path/to/other.pt  # optional ensemble detector
PLATE_CONF=0.28
PROCESS_EVERY_N=2
ROADLENS_PADDLE=1
OCR_LANG=eng
```

## Accuracy-critical camera setup

For a real deployment, model quality is only half the problem. Aim the camera so plates occupy enough pixels, minimize extreme perspective, use a fast shutter/appropriate IR at night, and avoid placing the camera where headlights directly saturate the sensor. A 4K camera poorly aimed at a distant road can be worse than a lower-resolution camera positioned correctly.

## Privacy / authorization

Only connect CCTV streams you are authorized to access and process plate data under an appropriate legal basis. Do not use RoadLens to access third-party camera systems without permission.
