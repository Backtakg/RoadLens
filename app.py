import os
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from flask import Flask, Response, jsonify, render_template, request
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

APP = Flask(__name__)
DB_PATH = os.getenv("ROADLENS_DB", "roadlens.db")
MODEL_REPO = os.getenv("PLATE_MODEL_REPO", "krishnamishra8848/Nepal-Vehicle-License-Plate-Detection")
MODEL_FILE = os.getenv("PLATE_MODEL_FILE", "last.pt")
MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "")
CONF = float(os.getenv("PLATE_CONF", "0.35"))
PROCESS_EVERY_N = max(1, int(os.getenv("PROCESS_EVERY_N", "2")))
OCR_LANG = os.getenv("OCR_LANG", "eng")

state = {
    "running": False,
    "source": None,
    "thread": None,
    "cap": None,
    "latest_jpeg": None,
    "lock": threading.Lock(),
    "last_error": None,
    "fps": 0.0,
    "frames": 0,
    "started_at": None,
    "recent_plates": deque(maxlen=80),
    "last_seen": {},
}


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            plate TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            image BLOB
        )"""
    )
    con.commit()
    return con


def load_model():
    path = MODEL_PATH
    if not path:
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return YOLO(path)


MODEL = None
MODEL_ERROR = None
try:
    MODEL = load_model()
except Exception as exc:
    MODEL_ERROR = str(exc)


def normalize_plate(text: str) -> str:
    text = text.replace("|", "1").replace("I", "1").replace("O", "0")
    text = re.sub(r"[^0-9A-Za-z\u0900-\u097F]", "", text).upper()
    return text[:24]


def preprocess_variants(crop):
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    scale = max(3.0, min(6.0, 1000 / max(1, w)))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    den = cv2.fastNlMeansDenoising(clahe, None, 7, 7, 21)
    sharp = cv2.addWeighted(den, 1.6, cv2.GaussianBlur(den, (0, 0), 1.2), -0.6, 0)
    thr = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 7)
    return [crop, cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)]


def ocr_plate(crop):
    candidates = []
    for img in preprocess_variants(crop):
        for psm in (7, 8, 13):
            cfg = f"--oem 3 --psm {psm}"
            try:
                txt = pytesseract.image_to_string(img, lang=OCR_LANG, config=cfg)
            except Exception:
                txt = ""
            txt = normalize_plate(txt)
            if len(txt) >= 2:
                candidates.append(txt)
    if not candidates:
        return "", 0.0
    counts = Counter(candidates)
    best, votes = counts.most_common(1)[0]
    score = min(0.99, 0.45 + 0.12 * votes + min(0.25, len(best) * 0.015))
    return best, score


def consensus_update(plate, source, confidence, evidence_jpeg):
    if not plate:
        return
    now = time.time()
    state["recent_plates"].append((now, plate, confidence))
    # Temporal voting: require repeated agreement before writing an event.
    window = [p for t, p, c in state["recent_plates"] if now - t < 4.0]
    votes = Counter(window)
    winner, count = votes.most_common(1)[0]
    if winner != plate or count < 2:
        return
    last = state["last_seen"].get(winner, 0)
    if now - last < 8:
        return
    state["last_seen"][winner] = now
    con = db()
    con.execute(
        "INSERT INTO events(ts, plate, confidence, source, image) VALUES(?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), winner, float(confidence), str(source), evidence_jpeg),
    )
    con.commit()
    con.close()


def process_frame(frame):
    annotated = frame.copy()
    if MODEL is None:
        cv2.putText(annotated, "MODEL ERROR - see /api/status", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return annotated
    try:
        results = MODEL.predict(frame, conf=CONF, verbose=False, imgsz=960, max_det=30)
        for result in results:
            if result.boxes is None:
                continue
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                plate, ocr_conf = ocr_plate(crop)
                label = f"{plate or 'plate'}  {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 30)), (x1 + min(420, 12 * len(label) + 20), y1), (0, 220, 120), -1)
                cv2.putText(annotated, label, (x1 + 8, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
                if plate:
                    ok, jpg = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    consensus_update(plate, state.get("source"), min(0.99, float(conf) * 0.6 + ocr_conf * 0.4), jpg.tobytes() if ok else None)
    except Exception as exc:
        state["last_error"] = str(exc)
    return annotated


def open_source(source):
    if source is None:
        return None
    s = str(source).strip()
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    return cv2.VideoCapture(s, cv2.CAP_FFMPEG)


def capture_loop():
    cap = open_source(state["source"])
    with state["lock"]:
        state["cap"] = cap
    if cap is None or not cap.isOpened():
        state["last_error"] = "Could not open source. For CCTV, use an authorized RTSP/HTTP stream reachable from this machine."
        state["running"] = False
        return
    prev = time.time()
    frames_since = 0
    while state["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.25)
            continue
        state["frames"] += 1
        frames_since += 1
        annotated = process_frame(frame) if frames_since % PROCESS_EVERY_N == 0 else frame
        ok, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with state["lock"]:
                state["latest_jpeg"] = jpg.tobytes()
        now = time.time()
        if now - prev >= 1:
            state["fps"] = frames_since / (now - prev)
            frames_since = 0
            prev = now
    cap.release()
    with state["lock"]:
        state["cap"] = None


@APP.get('/')
def index():
    return render_template('index.html')


@APP.get('/api/status')
def status():
    return jsonify({
        "running": state["running"],
        "source": state["source"],
        "fps": round(state["fps"], 1),
        "frames": state["frames"],
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
        "last_error": state["last_error"],
    })


@APP.post('/api/start')
def start():
    source = (request.json or {}).get('source', '0')
    if not source:
        return jsonify({"error": "source is required"}), 400
    if state["running"]:
        state["running"] = False
        if state["thread"]:
            state["thread"].join(timeout=2)
    state["source"] = source
    state["last_error"] = None
    state["frames"] = 0
    state["running"] = True
    state["started_at"] = datetime.now(timezone.utc).isoformat()
    state["thread"] = threading.Thread(target=capture_loop, daemon=True)
    state["thread"].start()
    return jsonify({"ok": True})


@APP.post('/api/stop')
def stop():
    state["running"] = False
    return jsonify({"ok": True})


@APP.get('/api/events')
def events():
    limit = min(200, max(1, int(request.args.get('limit', 50))))
    con = db()
    rows = con.execute(
        "SELECT id, ts, plate, confidence, source FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@APP.get('/video_feed')
def video_feed():
    def gen():
        while True:
            with state["lock"]:
                frame = state["latest_jpeg"]
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    db().close()
    APP.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), threaded=True)
