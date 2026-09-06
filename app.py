import csv
import io
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from roadlens_engine import ANPREngine

APP = Flask(__name__)
DB_PATH = os.getenv("ROADLENS_DB", "roadlens.db")
PROCESS_EVERY_N = max(1, int(os.getenv("PROCESS_EVERY_N", "2")))
CONSENSUS_SECONDS = float(os.getenv("CONSENSUS_SECONDS", "5"))
CONSENSUS_VOTES = max(2, int(os.getenv("CONSENSUS_VOTES", "3")))
EVENT_COOLDOWN = float(os.getenv("EVENT_COOLDOWN", "10"))

state = {"running": False, "source": None, "thread": None, "cap": None,
         "latest_jpeg": None, "lock": threading.Lock(), "last_error": None,
         "fps": 0.0, "frames": 0, "started_at": None, "tracks": {}, "last_seen": {}}
ENGINE = None
ENGINE_ERROR = None
try:
    ENGINE = ANPREngine()
except Exception as exc:
    ENGINE_ERROR = f"ANPR engine failed to initialize: {type(exc).__name__}: {exc}"


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, plate TEXT NOT NULL,
        confidence REAL NOT NULL, source TEXT NOT NULL, track_id INTEGER, image BLOB,
        vehicle_type TEXT, vehicle_brand TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(events)").fetchall()}
    for name, typ in (("track_id", "INTEGER"), ("vehicle_type", "TEXT"), ("vehicle_brand", "TEXT")):
        if name not in cols:
            con.execute(f"ALTER TABLE events ADD COLUMN {name} {typ}")
    con.commit()
    return con


def classify_vehicle(crop):
    # Vehicle make/type models are optional. Never infer a brand from a plate.
    # Return Unknown until a validated classifier is installed/configured.
    return "Unknown", "Unknown"


def record_consensus(plate, source, confidence, image, track_id=None, vehicle_type="Unknown", vehicle_brand="Unknown"):
    if not plate:
        return
    now = time.time()
    key = (str(source), int(track_id)) if track_id is not None else (str(source), "untracked")
    history = state["tracks"].setdefault(key, deque(maxlen=60))
    history.append((now, plate, float(confidence), vehicle_type, vehicle_brand))
    cutoff = now - CONSENSUS_SECONDS
    while history and history[0][0] < cutoff:
        history.popleft()
    votes = Counter(p for _, p, _, _, _ in history)
    winner, count = votes.most_common(1)[0]
    if winner != plate or count < CONSENSUS_VOTES:
        return
    matching = [(c, vt, vb) for t, p, c, vt, vb in history if p == winner and t >= cutoff]
    score = sum(c for c, _, _ in matching) / max(1, len(matching))
    vehicle_types = Counter(vt for _, vt, _ in matching if vt and vt != "Unknown")
    brands = Counter(vb for _, _, vb in matching if vb and vb != "Unknown")
    final_type = vehicle_types.most_common(1)[0][0] if vehicle_types else "Unknown"
    final_brand = brands.most_common(1)[0][0] if brands else "Unknown"
    event_key = (str(source), track_id, winner)
    last = state["last_seen"].get(event_key, 0)
    if now - last < EVENT_COOLDOWN:
        return
    state["last_seen"][event_key] = now
    con = db()
    con.execute("INSERT INTO events(ts,plate,confidence,source,track_id,image,vehicle_type,vehicle_brand) VALUES(?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), winner, score, str(source), track_id, image, final_type, final_brand))
    con.commit()
    con.close()


def process_frame(frame):
    annotated = frame.copy()
    if ENGINE is None:
        cv2.putText(annotated, "ANPR ENGINE UNAVAILABLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return annotated
    try:
        for box, det_conf, track_id in ENGINE.detect(frame):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            plate, ocr_conf, _ = ENGINE.recognize(crop)
            total = (0.55 * det_conf + 0.45 * ocr_conf) if plate else det_conf
            vehicle_type, vehicle_brand = classify_vehicle(crop)
            label = f"{plate or 'PLATE'} | {vehicle_type} | {vehicle_brand} | {total:.2f}"
            if track_id is not None:
                label += f" #{track_id}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 32)), (x1 + min(700, 13 * len(label) + 24), y1), (0, 220, 120), -1)
            cv2.putText(annotated, label, (x1 + 8, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)
            if plate:
                ok, jpg = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
                record_consensus(plate, state["source"], total, jpg.tobytes() if ok else None, track_id, vehicle_type, vehicle_brand)
    except Exception as exc:
        state["last_error"] = f"Frame processing failed: {type(exc).__name__}: {exc}"
    return annotated


def open_source(source):
    s = str(source or "0").strip()
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    if not s.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        raise ValueError("Source must be a webcam index or an RTSP/RTSPS/HTTP(S) URL")
    return cv2.VideoCapture(s, cv2.CAP_FFMPEG)


def capture_loop():
    try:
        cap = open_source(state["source"])
    except Exception as exc:
        state["last_error"] = str(exc); state["running"] = False; return
    with state["lock"]:
        state["cap"] = cap
    if cap is None or not cap.isOpened():
        state["last_error"] = "Could not open source. Verify the camera is reachable and the URL is correct."; state["running"] = False
        if cap is not None: cap.release()
        return
    prev, processed = time.time(), 0
    try:
        while state["running"]:
            ok, frame = cap.read()
            if not ok:
                state["last_error"] = "Camera read failed; stream may be offline or disconnected."; time.sleep(0.15); continue
            state["last_error"] = None; state["frames"] += 1; processed += 1
            annotated = process_frame(frame) if processed % PROCESS_EVERY_N == 0 else frame
            ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with state["lock"]: state["latest_jpeg"] = jpg.tobytes()
            now = time.time()
            if now - prev >= 1.0:
                state["fps"] = processed / (now - prev); processed, prev = 0, now
    finally:
        cap.release()
        with state["lock"]: state["cap"] = None


@APP.get("/")
def index(): return render_template("index.html")


@APP.get("/api/health")
def health():
    return jsonify({"ok": ENGINE is not None, "engine": "ready" if ENGINE else "unavailable", "error": ENGINE_ERROR}), (200 if ENGINE else 503)


@APP.get("/api/status")
def status():
    return jsonify({"running": state["running"], "source": state["source"], "fps": round(state["fps"], 1), "frames": state["frames"],
                    "model_loaded": ENGINE is not None, "model_error": ENGINE_ERROR, "last_error": state["last_error"],
                    "consensus": {"seconds": CONSENSUS_SECONDS, "votes": CONSENSUS_VOTES},
                    "ocr_backends": {"tesseract": True, "paddleocr": bool(ENGINE and ENGINE.paddle is not None), "second_detector": bool(ENGINE and ENGINE.detector2 is not None)},
                    "vehicle_classifier": "not_configured", "vehicle_brand": "not_configured"})


@APP.post("/api/start")
def start():
    source = str((request.json or {}).get("source", "0")).strip()
    if not source: return jsonify({"error": "source is required"}), 400
    if ENGINE is None: return jsonify({"error": "ANPR engine is unavailable", "detail": ENGINE_ERROR}), 503
    if state["running"]:
        state["running"] = False
        if state["thread"]: state["thread"].join(timeout=2)
    state.update({"source": source, "last_error": None, "frames": 0, "fps": 0.0, "running": True, "started_at": datetime.now(timezone.utc).isoformat(), "latest_jpeg": None})
    state["tracks"].clear(); state["last_seen"].clear()
    state["thread"] = threading.Thread(target=capture_loop, daemon=True); state["thread"].start()
    return jsonify({"ok": True, "source": source})


@APP.post("/api/stop")
def stop(): state["running"] = False; return jsonify({"ok": True})


@APP.get("/api/events")
def events():
    try: limit = min(200, max(1, int(request.args.get("limit", 50))))
    except ValueError: return jsonify({"error": "limit must be an integer"}), 400
    con = db(); rows = con.execute("SELECT id,ts,plate,confidence,source,track_id,vehicle_type,vehicle_brand FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall(); con.close()
    return jsonify([dict(r) for r in rows])


@APP.get("/api/export.csv")
def export_csv():
    con = db(); rows = con.execute("SELECT ts,plate,vehicle_type,vehicle_brand,confidence,source,track_id FROM events ORDER BY id ASC").fetchall(); con.close()
    out = io.StringIO(); writer = csv.writer(out); writer.writerow(["Date", "Time", "Plate Number", "Vehicle Type", "Vehicle Brand", "Confidence", "Source", "Track ID"])
    for r in rows:
        dt = datetime.fromisoformat(r["ts"]); writer.writerow([dt.date().isoformat(), dt.time().replace(microsecond=0).isoformat(), r["plate"], r["vehicle_type"] or "Unknown", r["vehicle_brand"] or "Unknown", r["confidence"], r["source"], r["track_id"]])
    data = io.BytesIO(out.getvalue().encode("utf-8-sig")); return send_file(data, mimetype="text/csv", as_attachment=True, download_name="roadlens_detections.csv")


@APP.get("/video_feed")
def video_feed():
    def gen():
        while True:
            with state["lock"]: frame, running = state["latest_jpeg"], state["running"]
            if frame: yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
            elif not running: time.sleep(0.15)
            else: time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    db().close(); APP.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), threaded=True)
