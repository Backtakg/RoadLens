import os
import re
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

MODEL_REPO = os.getenv("PLATE_MODEL_REPO", "krishnamishra8848/Nepal-Vehicle-License-Plate-Detection")
MODEL_FILE = os.getenv("PLATE_MODEL_FILE", "last.pt")
MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "")
MODEL_PATH_2 = os.getenv("PLATE_MODEL_PATH_2", "")
CONF = float(os.getenv("PLATE_CONF", "0.28"))


def load_detector(path=None):
    if path:
        return YOLO(path)
    return YOLO(MODEL_PATH or hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


class ANPREngine:
    def __init__(self):
        self.detector = load_detector()
        self.detector2 = load_detector(MODEL_PATH_2) if MODEL_PATH_2 else None
        self.paddle = None
        if PaddleOCR is not None and os.getenv("ROADLENS_PADDLE", "1") == "1":
            try:
                self.paddle = PaddleOCR(lang="ne", use_doc_orientation_classify=False,
                                        use_doc_unwarping=False, use_textline_orientation=False)
            except Exception:
                self.paddle = None

    def detect(self, frame):
        boxes = []
        for model in (self.detector, self.detector2):
            if model is None:
                continue
            try:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     conf=CONF, iou=0.55, imgsz=960, max_det=50, verbose=False)[0]
            except Exception:
                result = model.predict(frame, conf=CONF, imgsz=960, max_det=50, verbose=False)[0]
            if result.boxes is None:
                continue
            ids = result.boxes.id.int().cpu().tolist() if result.boxes.is_track else [None] * len(result.boxes)
            for b, c, tid in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), ids):
                boxes.append((b.astype(int), float(c), tid))

        # A classical OpenCV proposal stage is only a fallback when learned detectors miss.
        if not boxes:
            for b, score in classical_plate_candidates(frame):
                boxes.append((b, score, None))
        return self._merge_boxes(boxes)

    @staticmethod
    def _merge_boxes(boxes, iou_threshold=0.55):
        kept = []
        for candidate in sorted(boxes, key=lambda x: x[1], reverse=True):
            if all(iou(candidate[0], old[0]) < iou_threshold for old in kept):
                kept.append(candidate)
        return kept[:30]

    def recognize(self, crop):
        candidates = []
        for image in preprocess_variants(crop):
            candidates.extend(tesseract_candidates(image))
            if self.paddle is not None:
                candidates.extend(paddle_candidates(self.paddle, image))
        candidates = [(normalize_plate(t), float(s)) for t, s in candidates if valid_candidate(normalize_plate(t))]
        if not candidates:
            return "", 0.0, []
        score_by_text = defaultdict(float)
        evidence = defaultdict(list)
        for text, score in candidates:
            score_by_text[text] += max(0.05, score)
            evidence[text].append(score)
        text = max(score_by_text, key=score_by_text.get)
        confidence = min(0.995, 0.35 + 0.08 * len(evidence[text]) + 0.12 * max(evidence[text]))
        return text, confidence, candidates


def classical_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    proposals = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.0002 * w * h or area > 0.12 * w * h:
            continue
        ratio = cw / max(1, ch)
        if not 1.8 <= ratio <= 7.5:
            continue
        rectangularity = cv2.contourArea(c) / max(1, area)
        if rectangularity < 0.35:
            continue
        score = min(0.55, 0.25 + 0.25 * rectangularity + 0.05 * min(2.0, ratio / 3.0))
        proposals.append((np.array([x, y, x + cw, y + ch]), score))
    return sorted(proposals, key=lambda z: z[1], reverse=True)[:10]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(area_a + area_b - inter)


def normalize_plate(text):
    text = str(text or "").replace("|", "1").replace("I", "1").replace("O", "0")
    return re.sub(r"[^0-9A-Za-z\u0900-\u097F]", "", text).upper()[:24]


def valid_candidate(text):
    return 3 <= len(text) <= 16 and any(ch.isdigit() for ch in text) and any(ch.isalpha() or "\u0900" <= ch <= "\u097F" for ch in text)


def preprocess_variants(crop):
    if crop is None or crop.size == 0:
        return []
    _, w = crop.shape[:2]
    scale = max(3.0, min(7.0, 1200.0 / max(1, w)))
    base = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    den = cv2.fastNlMeansDenoising(clahe, None, 7, 7, 21)
    sharp = cv2.addWeighted(den, 1.7, cv2.GaussianBlur(den, (0, 0), 1.1), -0.7, 0)
    variants = [base, cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)]
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        thr = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, 5)
        variants.append(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    for gamma in (0.65, 1.5):
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        variants.append(cv2.LUT(base, lut))
    return variants


def tesseract_candidates(image):
    out = []
    for lang in (os.getenv("OCR_LANG", "nep+eng"), "eng"):
        for psm in (6, 7, 8, 11, 13):
            try:
                data = pytesseract.image_to_data(image, lang=lang, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT)
                for txt, conf in zip(data["text"], data["conf"]):
                    txt = normalize_plate(txt)
                    if txt:
                        try:
                            score = max(0.0, float(conf) / 100.0)
                        except Exception:
                            score = 0.25
                        out.append((txt, score))
            except Exception:
                pass
    return out


def paddle_candidates(paddle, image):
    out = []
    try:
        results = paddle.predict(input=image)
        for result in results:
            payload = result.json if hasattr(result, "json") else result
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                continue
            values = payload.get("rec_texts", payload.get("text", payload.get("texts", [])))
            scores = payload.get("rec_scores", payload.get("scores", []))
            if isinstance(values, list):
                for i, text in enumerate(values):
                    score = float(scores[i]) if i < len(scores) else 0.5
                    out.append((normalize_plate(text), score))
    except Exception:
        pass
    return out
