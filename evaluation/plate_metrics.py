"""Truthful, offline evaluation utilities for RoadLens.

This module computes metrics from a user-supplied ground-truth CSV and RoadLens
predictions. It never invents or fills in missing measurements.

CSV columns:
    image_id,ground_truth,prediction

A prediction may be empty for a missed read. Both strings are normalized using
RoadLens's plate normalization rules before scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from roadlens_engine import normalize_plate


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(rows):
    total = len(rows)
    exact = 0
    missed = 0
    errors = 0
    gt_chars = 0
    for row in rows:
        gt = normalize_plate(row.get("ground_truth", ""))
        pred = normalize_plate(row.get("prediction", ""))
        if not pred:
            missed += 1
        if gt == pred and gt:
            exact += 1
        errors += edit_distance(gt, pred)
        gt_chars += len(gt)
    return {
        "samples": total,
        "exact_plate_accuracy": exact / total if total else None,
        "missed_read_rate": missed / total if total else None,
        "character_error_rate": errors / gt_chars if gt_chars else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoadLens plate reads against ground truth.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics = evaluate(rows)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
