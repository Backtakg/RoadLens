# RoadLens real-world evaluation

RoadLens does not display a fabricated accuracy number. A metric appears only after it has been computed from labeled, held-out data.

## Ground-truth CSV

Create a CSV with:

```csv
image_id,ground_truth,prediction
frame_0001,BA123PA4567,BA123PA4567
frame_0002,BA123PA4567,BA123PA4561
frame_0003,BA998AA1122,
```

The examples above are only the **format**; they are not benchmark data and must not be interpreted as RoadLens results.

## Required test conditions

Keep separate labeled test subsets for:

- daylight
- night
- rain
- fog/haze
- glare/headlights
- motion blur

If a condition has no labeled samples, report `not measured` rather than `0%` or an estimate.

## Metrics

Run:

```bash
python evaluation/plate_metrics.py path/to/results.csv --out evaluation/metrics.json
```

Report:

- `samples`
- `exact_plate_accuracy`
- `missed_read_rate`
- `character_error_rate`

For detector evaluation, also report precision, recall, mAP50 and mAP50-95 from the held-out detector test split. Detection metrics and OCR metrics must not be conflated.

## Production benchmark protocol

For each authorized camera, record camera ID, resolution, frame rate, viewpoint, approximate plate distance, date/time range, lighting/weather condition, plate type and readable/unreadable count. Freeze the test set before tuning thresholds. Keep vehicle/plate identities separated between train and test where possible to reduce leakage.

A model is not considered production-validated merely because its training loss decreases or because a public model reports high metrics on a different dataset.

## Confidence warning

A detector/OCR confidence score is **not an accuracy percentage**. RoadLens must never convert confidence into a claimed accuracy. The live application labels this value as `conf` and the database stores it only as an uncalibrated confidence signal.
