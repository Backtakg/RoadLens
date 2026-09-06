# RoadLens training plan

RoadLens is deliberately data-centric: the model should be retrained on Nepal plates and then adapted to the actual camera geometry and weather of deployment.

## Public data used

1. **Nepali Private License Plates** — `mukulboro/nepali-private-license-plates`
   - CC BY 4.0
   - 1,172 source photographs
   - 5,860 image/label pairs including blur, contrast, exposure and noise variants
   - 3 Nepal plate-type classes: embossed, provincial, regional
   - ~810 MB

2. **Nepali number-plate character research**
   - The June 2026 paper *Character Recognition of Nepali Number Plate* reports a 34-character Devanagari plate-character dataset with 26,537 labeled samples and a YOLO + CNN recognition pipeline reaching up to 93% character recognition accuracy.
   - RoadLens does not bundle this data because its distribution terms should be checked before redistribution. Use it locally where its terms permit.

3. **Nepali LPR / character resources**
   - `Prasanna1991/LPR` and `Prasanna1991/DHCD_Dataset` are useful research resources for Nepali plate/Devanagari character recognition.

## Why we do not simply download everything and claim accuracy

A random internet scrape can contain duplicates, incompatible licenses, incorrect labels, or images from cameras that look nothing like your CCTV. RoadLens therefore separates:

**public pretraining → Nepal fine-tuning → camera-specific calibration → held-out evaluation**

The detector training script splits the Nepali private-plate dataset by source-family so an original image and its blur/contrast/noise copies do not leak across train and validation/test.

## Free training

The intended zero-cost route is a free GPU notebook such as Google Colab or Kaggle:

```bash
pip install -r requirements.txt
python training/train_nepal_detector.py --epochs 50 --device 0
```

Then point RoadLens at the resulting model:

```bash
PLATE_MODEL_PATH=training/runs/nepal_plate_detector/weights/best.pt python app.py
```

For a second detector, train a different architecture/checkpoint and set:

```bash
PLATE_MODEL_PATH_2=/path/to/second/best.pt
```

RoadLens will ensemble both detectors and suppress duplicate overlapping boxes.

## Weather-specific adaptation

For the final deployment set, collect **authorized** frames from the exact camera under:

- clear daylight
- overcast
- rain/water droplets
- night/low exposure
- headlights/glare
- fog/haze
- motion blur
- compression/low bitrate
- motorcycles, cars, buses and trucks
- near/far plate sizes and oblique angles

Never put those frames into the test split until after model development. The test set should remain unseen so the measured accuracy means something.

## Metrics RoadLens should report

- Plate detection: precision, recall, mAP50 and mAP50-95
- Plate-level OCR: exact-match accuracy
- Character error rate (CER)
- False-read rate per 1,000 vehicles
- Missed-plate rate
- Track-level exact-match accuracy
- Results broken down by day/night/rain/fog/vehicle type

A deployment should not be called "accurate" merely because a model's public benchmark is high.
