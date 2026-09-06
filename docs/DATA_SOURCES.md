# RoadLens data and model provenance

This file separates **external published claims** from **RoadLens measurements**.

## 1. Nepal plate detection dataset

Source: https://huggingface.co/datasets/mukulboro/nepali-private-license-plates

The dataset card states that it contains 1,172 unique source photos and 5,860 image/label pairs, with original, blur, contrast, exposure and noise variants. It is released under CC BY 4.0. The dataset card also states that permission was obtained for the source parking-space images.

These are dataset facts, not RoadLens accuracy results.

## 2. Existing Nepal plate detector

Source: https://huggingface.co/krishnamishra8848/Nepal-Vehicle-License-Plate-Detection

The model card reports Precision 0.973, Recall 0.956, mAP@50 0.988 and mAP@50-95 0.929 for its own evaluation. RoadLens does **not** copy those values into its own accuracy dashboard.

## 3. Nepal character detector research/model

Source: https://huggingface.co/krishnamishra8848/Nepal_Vehicle_License_Plates_Detection_Version3

The publisher describes this as a character-wise Nepal plate detector and reports its own evaluation metrics. Those results are external to RoadLens and must not be represented as RoadLens results.

## 4. OCR

PaddleOCR documentation: https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html

The official documentation lists Nepali (`ne`) among the supported PP-OCRv5 languages.

## 5. Tracking

Ultralytics documentation: https://docs.ultralytics.com/modes/track

RoadLens can use ByteTrack for persistent object/plate tracks. Tracking capability is an implementation feature; it is not an accuracy claim.

## RoadLens benchmark status

Until a labeled, held-out RoadLens test set has actually been evaluated, RoadLens has **no validated deployment accuracy number**. The repository's evaluation tooling is designed to produce those measurements from real labeled data.
