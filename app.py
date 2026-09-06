import os
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import cv2
from flask import Flask, Response, jsonify, render_template, request

from roadlens_engine import ANPREngine

APP = Flask(__name__)
DB_PATH = os.getenv("ROADLENS_DB", "roadlens.db")
PROCESS_EVERY_N = max(1, int(os.getenv("PROCESS_EVERY_N", "2")))

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
    "recent": deque(maxlen=120),
    "last_seen": {},
}

ENGINE = None
ENGINE_ERROR = None
try:
    ENGINE = ANPREngine()
except Exception as exc:
    ENGINE_ERROR = str(exc)


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        plate TEXT NOT NULL,
        confidence REAL NOT NULL,
        source TEXT NOT NULL,
        image BLOB
    )""")
    con.commit()
    return con


def record_consensus(plate, source, confidence, image):
    if not plate:
        return
    now = time.time()
    state["recent"].append((now, plate, confidence))
    window = [(p, c) for t, p, c in state["recent"] if now - t <= 5.0]
    if not window:
        return
    votes = Counter(p for p, _ in window)
    winner, count = votes.most_common(1)[0]
    if winner != plate or count < 3:
        return
    scores = [c for p, c in window if p == winner]
    score = min(0.995, sum(scores) / max(1, len(scores)) + 0.03 * min(3, count - 1))
    last = state["last_seen"].get(winner, 0)
    if now - last < 10:
        return
    state["last_seen"][winner] = now
    con = db()
    con.execute("INSERT INTO events(ts,plate,confidence,source,image) VALUES(?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), winner, score, str(source), image))
    con.commit()
    con.close()


def process_frame(frame):
    annotated = frame.copy()
    if ENGINE is None:
        cv2.putText(annotated, "ENGINE ERROR - see /api/status", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return annotated
    try:
        for box, det_conf, track_id in ENGINE.detect(frame):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            plate, ocr_conf, alternatives = ENGINE.recognize(crop)
            total = min(0.995, 0.55 * det_conf + 0.45 * ocr_conf) if plate else det_conf
            label = f"{plate or 'PLATE'}  {total:.2f}"
            if track_id is not None:
                label += f"  #{track_id}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 32)),
                           (x1 + min(520, 13 * len(label) + 24), y1), (0, 220, 120), -1)
            cv2.putText(annotated, label, (x1 + 8, y1 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            if plate:
                ok, jpg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                record_consensus(plate, state["source"], total, jpg.tobytes() if ok else None)
    except Exception as exc:
        state["last_error"] = str(exc)
    return annotated


def open_source(source):
    s = str(source or "0").strip()
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    return cv2.VideoCapture(s, cv2.CAP_FFMPEG)


def capture_loop():
    cap = open_source(state["source"])
    with state["lock"]:
        state["cap"] = cap
    if cap is None or not cap.isOpened():
        state["last_error"] = "Could not open source. Use a local camera or an authorized RTSP/HTTP stream reachable from this machine."
        state["running"] = False
        return
    prev = time.time()
    processed = 0
    while state["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.15)
            continue
        state["frames"] += 1
        processed += 1
        annotated = process_frame(frame) if processed % PROCESS_EVERY_N == 0 else frame
        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with state["lock"]:
                state["latest_jpeg"] = jpg.tobytes()
        now = time.time()
        if now - prev >= 1.0:
            state["fps"] = processed / (now - prev)
            processed = 0
            prev = now
    cap.release()
    with state["lock"]:
        state["cap"] = None


@APP.get("/")
def index():
    return render_template("index.html")


@APP.get("/api/status")
def status():
    return jsonify({
        "running": state["running"],
        "source": state["source"],
        "fps": round(state["fps"], 1),
        "frames": state["frames"],
        "model_loaded": ENGINE is not None,
        "model_error": ENGINE_ERROR,
        "last_error": state["last_error"],
        "ocr_backends": {
            "tesseract": True,
            "paddleocr": bool(ENGINE and ENGINE.paddle is not None),
            "second_detector": bool(ENGINE and ENGINE.detector2 is not None),
        },
    })


@APP.post("/api/start")
def start():
    source = (request.json or {}).get("source", "0")
    if not source:
        return jsonify({"error": "source is required"}), 400
    if state["running"]:
        state["running"] = False
        if state["thread"]:
            state["thread"].join(timeout=2)
    state["source"] = source
    state["last_error"] = None
    state["frames"] = 0
    state["recent"].clear()
    state["running"] = True
    state["started_at"] = datetime.now(timezone.utc).isoformat()
    state["thread"] = threading.Thread(target=capture_loop, daemon=True)
    state["thread"].start()
    return jsonify({"ok": True})


@APP.post("/api/stop")
def stop():
    state["running"] = False
    return jsonify({"ok": True})


@APP.get("/api/events")
def events():
    limit = min(200, max(1, int(request.args.get("limit", 50))))
    con = db()
    rows = con.execute("SELECT id,ts,plate,confidence,source FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@APP.get("/video_feed")
def video_feed():
    def gen():
        while True:
            with state["lock"]:
                frame = state["latest_jpeg"]
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    db().close()
    APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
