# RoadLens — Nepal ANPR

RoadLens is a **zero-paid-API-cost, self-hosted** Automatic Number Plate Recognition starter designed around Nepal vehicle plates and authorized CCTV/RTSP feeds.

## What is included

- YOLO-based Nepal plate detector loaded from an open Hugging Face model.
- Live RTSP/HTTP camera ingestion with OpenCV/FFmpeg.
- Local webcam support.
- Plate crop enhancement for difficult lighting: CLAHE, denoising, sharpening and adaptive thresholding.
- Tesseract OCR with `nep+eng` in the Docker image.
- Multi-frame voting so one bad OCR frame is less likely to become an event.
- SQLite event history.
- Browser dashboard showing the live annotated feed and detected plates.
- No paid cloud API is required; inference and storage happen on the machine running RoadLens.

## Run with Docker

```bash
docker compose up --build
```

Open `http://localhost:5000`.

For a local USB camera, choose **Local camera** and use `0` (or another camera index).

For an authorized CCTV camera, choose **RTSP / CCTV** and enter the RTSP URL, for example:

```text
rtsp://user:password@192.168.1.50:554/stream
```

The browser does not need direct RTSP access: the RoadLens machine opens the RTSP stream and sends an annotated MJPEG feed to the dashboard.

## Run without Docker

Install Python dependencies from `requirements.txt`, make sure OpenCV can access your camera/stream and Tesseract is installed with the Nepali language pack, then:

```bash
python app.py
```

## Important accuracy note

There is no honest way to promise **100% accuracy in every weather condition**. Rain, fog, glare, darkness, motion blur, compression, plate angle and insufficient plate pixels can make any ANPR system fail. The current design improves robustness with image preprocessing and temporal consensus, but you should benchmark it on the exact CCTV cameras and locations where you intend to deploy it.

For a production-grade Nepal deployment, the next step is to train/fine-tune a detector and OCR model on **your actual camera footage**, including daytime, nighttime, rain, fog, glare, motorcycles, buses, trucks and different plate formats.

## Model/data notes

The default detector is the public `krishnamishra8848/Nepal-Vehicle-License-Plate-Detection` model. Its model card reports strong validation metrics, but those metrics are not a guarantee for your cameras or field conditions.

The project can be pointed at your own YOLO weights with:

```bash
PLATE_MODEL_PATH=/path/to/best.pt python app.py
```

or by changing `PLATE_MODEL_REPO` and `PLATE_MODEL_FILE`.

## CCTV/privacy

Only connect cameras and process plate data where you have authorization and an appropriate legal basis. Do not use RoadLens to access or collect feeds you are not authorized to access.
