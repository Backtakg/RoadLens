# RoadLens evaluation protocol

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

## Run

```bash
python evaluation/plate_metrics.py path/to/results.csv --out evaluation/metrics.json
```

The script reports:

- `samples`
- `exact_plate_accuracy`
- `missed_read_rate`
- `character_error_rate`

For a deployment benchmark, keep the test set completely separate from training data. Report results separately for day/night/rain/fog/glare where the test set contains enough labeled samples. Do not combine those conditions into one number and call it "all-weather accuracy."

## Required production benchmark

For each camera, collect an authorized and legally appropriate labeled set containing the real viewpoints and conditions that matter. Freeze the test set before tuning thresholds. Keep vehicle/plate identities separated between train and test where possible to reduce leakage. Record the camera, date range, lighting/weather condition, plate type, and number of readable/unreadable plates.

A model is not considered production-validated merely because its training loss decreases or because a public model reports high metrics on a different dataset.
