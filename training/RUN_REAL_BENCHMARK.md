# Run the real RoadLens benchmark

RoadLens does **not** contain fabricated accuracy numbers. This procedure creates measured results from a real training run and a held-out test split.

## Free GPU route

Use a free GPU notebook (Colab/Kaggle) or a machine with a CUDA-capable GPU.

```bash
git clone https://github.com/Backtakg/RoadLens.git
cd RoadLens
pip install -r requirements.txt
python training/benchmark.py --device 0 --epochs 50 --imgsz 960 --batch 16
```

The script:

1. Downloads the public `mukulboro/nepali-private-license-plates` dataset.
2. Splits augmentation families together so variants of one source do not leak across splits.
3. Fine-tunes the detector when `--weights` is not supplied.
4. Evaluates the resulting weights on `split=test`, separately from training.
5. Writes **measured** precision, recall, mAP50 and mAP50-95 to `training/runs/nepal_plate_detector/test_metrics.json`.

To benchmark an already trained checkpoint without retraining:

```bash
python training/benchmark.py \
  --weights training/runs/nepal_plate_detector/weights/best.pt \
  --device 0
```

## Important

The JSON file is intentionally not committed as a claimed result until the command has actually been executed. A number is considered a RoadLens result only when it comes from this benchmark (or another documented held-out test) and includes its dataset, split, weights and timestamp.

Ultralytics validation reports precision, recall, mAP50 and mAP50-95 from labeled ground truth. For a held-out test measurement, the dataset must define a test split and validation must be run with `split="test"`.

## Deployment benchmark

Public-dataset metrics are detector metrics only. They do not establish live CCTV plate-reading accuracy. After detector training, create an authorized, manually labeled CCTV test set and report separately:

- exact plate match rate
- character error rate
- missed reads
- false reads per 1,000 vehicles
- day/night/rain/fog/glare breakdown
- camera-by-camera results

Never replace these measurements with a public model's reported score.
