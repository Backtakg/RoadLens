from evaluation.plate_metrics import edit_distance, evaluate


def test_edit_distance():
    assert edit_distance("ABC", "ABC") == 0
    assert edit_distance("ABC", "ABD") == 1
    assert edit_distance("ABC", "") == 3


def test_evaluate_exact_missed_and_cer():
    rows = [
        {"ground_truth": "BA123PA4567", "prediction": "BA123PA4567"},
        {"ground_truth": "BA123PA4567", "prediction": "BA123PA4561"},
        {"ground_truth": "BA998AA1122", "prediction": ""},
    ]
    metrics = evaluate(rows)
    assert metrics["samples"] == 3
    assert metrics["exact_plate_accuracy"] == 1 / 3
    assert metrics["missed_read_rate"] == 1 / 3
    # 1 substitution + 11 deletions over 33 ground-truth characters.
    assert metrics["character_error_rate"] == 13 / 33
