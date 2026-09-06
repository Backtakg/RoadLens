import os
import re
from collections import defaultdict
import cv2
import numpy as np
import pytesseract
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

MODEL_REPO=os.getenv('PLATE_MODEL_REPO','krishnamishra8848/Nepal-Vehicle-License-Plate-Detection')
MODEL_FILE=os.getenv('PLATE_MODEL_FILE','last.pt')
MODEL_PATH=os.getenv('PLATE_MODEL_PATH','')
MODEL_PATH_2=os.getenv('PLATE_MODEL_PATH_2','')
CONF=float(os.getenv('PLATE_CONF','0.28'))

def load_detector(path=None):
    return YOLO(path) if path else YOLO(MODEL_PATH or hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILE))

class ANPREngine:
    def __init__(self):
        self.detector=load_detector(); self.detector2=load_detector(MODEL_PATH_2) if MODEL_PATH_2 else None
        self.paddle=None
        if os.getenv('ROADLENS_PADDLE','0')=='1':
            try:
                from paddleocr import PaddleOCR
                self.paddle=PaddleOCR(ocr_version='PP-OCRv5',lang='en',use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
            except Exception: self.paddle=None
        self.ocr_available=self._tesseract_available() or self.paddle is not None
        if not self.ocr_available: raise RuntimeError('No OCR backend is available')
    @staticmethod
    def _tesseract_available():
        try: return bool(pytesseract.get_tesseract_version())
        except Exception: return False
    def detect(self,frame):
        boxes=[]
        for model in (self.detector,self.detector2):
            if model is None: continue
            try: result=model.track(frame,persist=True,tracker='bytetrack.yaml',conf=CONF,iou=.55,imgsz=640,max_det=30,verbose=False)[0]
            except Exception: result=model.predict(frame,conf=CONF,imgsz=640,max_det=30,verbose=False)[0]
            if result.boxes is None: continue
            ids=result.boxes.id.int().cpu().tolist() if result.boxes.is_track else [None]*len(result.boxes)
            for b,c,tid in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),ids): boxes.append((b.astype(int),float(c),tid))
        if not boxes:
            boxes=[(b,s,None) for b,s in classical_plate_candidates(frame)]
        return self._merge_boxes(boxes)
    @staticmethod
    def _merge_boxes(boxes,iou_threshold=.55):
        kept=[]
        for candidate in sorted(boxes,key=lambda x:x[1],reverse=True):
            if all(iou(candidate[0],old[0])<iou_threshold for old in kept): kept.append(candidate)
        return kept[:30]
    def recognize(self,crop):
        candidates=[]
        for image in preprocess_variants(crop):
            if self._tesseract_available(): candidates.extend(tesseract_candidates(image))
            if self.paddle is not None: candidates.extend(paddle_candidates(self.paddle,image))
        cleaned=[]
        for text,score in candidates:
            text=normalize_plate(text)
            if valid_candidate(text): cleaned.append((text,float(score)))
        if not cleaned:return '',0.0,[]
        score_by_text=defaultdict(float); evidence=defaultdict(list)
        for text,score in cleaned: score_by_text[text]+=max(.05,score); evidence[text].append(score)
        text=max(score_by_text,key=score_by_text.get)
        confidence=min(.995,.35+.08*len(evidence[text])+.12*max(evidence[text]))
        return text,confidence,cleaned

def classical_plate_candidates(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(5,5),0); edges=cv2.Canny(gray,80,180); kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(17,5)); edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations=2); contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); h,w=frame.shape[:2]; proposals=[]
    for c in contours:
        x,y,cw,ch=cv2.boundingRect(c); area=cw*ch
        if area<.0002*w*h or area>.12*w*h: continue
        ratio=cw/max(1,ch); rectangularity=cv2.contourArea(c)/max(1,area)
        if not 1.8<=ratio<=7.5 or rectangularity<.35: continue
        score=min(.55,.25+.25*rectangularity+.05*min(2,ratio/3)); proposals.append((np.array([x,y,x+cw,y+ch]),score))
    return sorted(proposals,key=lambda z:z[1],reverse=True)[:10]

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b; ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2); inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter:return 0.
    return inter/float(max(1,ax2-ax1)*max(1,ay2-ay1)+max(1,bx2-bx1)*max(1,by2-by1)-inter)

def normalize_plate(text):
    text=str(text or '').replace('|','1').replace('I','1').replace('O','0'); return re.sub(r'[^0-9A-Za-z\u0900-\u097F]','',text).upper()[:24]

def valid_candidate(text): return 3<=len(text)<=16 and any(c.isdigit() for c in text) and any(c.isalpha() or '\u0900'<=c<='\u097F' for c in text)

def preprocess_variants(crop):
    if crop is None or crop.size==0:return []
    h,w=crop.shape[:2]; scale=max(2.,min(5.,1000./max(1,w))); base=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC); gray=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY); clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(gray); sharp=cv2.addWeighted(clahe,1.6,cv2.GaussianBlur(clahe,(0,0),1.1),-.6,0); variants=[base,cv2.cvtColor(clahe,cv2.COLOR_GRAY2BGR),cv2.cvtColor(sharp,cv2.COLOR_GRAY2BGR)]
    for mode in (cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV): variants.append(cv2.cvtColor(cv2.adaptiveThreshold(sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,mode,31,5),cv2.COLOR_GRAY2BGR))
    return variants

def tesseract_candidates(image):
    out=[]
    for lang in (os.getenv('OCR_LANG','eng'),):
        for psm in (6,7,8,11,13):
            try:
                data=pytesseract.image_to_data(image,lang=lang,config=f'--oem 3 --psm {psm}',output_type=pytesseract.Output.DICT)
                for txt,conf in zip(data['text'],data['conf']):
                    if txt.strip():
                        try: score=max(0,float(conf)/100)
                        except Exception: score=.25
                        out.append((txt,score))
            except Exception: pass
    return out

def paddle_candidates(paddle,image):
    out=[]
    try:
        for result in paddle.predict(input=image):
            payload=result.json() if hasattr(result,'json') and callable(result.json) else result.json if hasattr(result,'json') else result
            if isinstance(payload,str):
                import json; payload=json.loads(payload)
            if not isinstance(payload,dict):continue
            texts=payload.get('rec_texts',payload.get('text',[])); scores=payload.get('rec_scores',payload.get('scores',[]))
            for i,text in enumerate(texts if isinstance(texts,list) else []): out.append((text,float(scores[i]) if i<len(scores) else .5))
    except Exception: pass
    return out
