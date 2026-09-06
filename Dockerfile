FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tesseract-ocr tesseract-ocr-nep \
    libgl1 libglib2.0-0 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep the Render web service within the 512 MiB class of instances.
# PaddlePaddle/PaddleOCR is an optional heavy backend and is disabled on the
# web deployment; Tesseract + the configured YOLO detector remain available.
COPY . .

ENV PORT=5000
ENV ROADLENS_PADDLE=0
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 5000
CMD ["python", "app.py"]
