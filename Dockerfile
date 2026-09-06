FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg tesseract-ocr tesseract-ocr-nep \
    libgl1 libglib2.0-0 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Open-source local OCR ensemble. No API key and no paid service.
RUN python -m pip install --no-cache-dir paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && pip install --no-cache-dir "paddleocr>=3.0"

COPY . .

ENV PORT=5000
ENV ROADLENS_PADDLE=1
EXPOSE 5000
CMD ["python", "app.py"]
