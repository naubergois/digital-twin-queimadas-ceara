"""Limiar adaptativo por dia (grade real vs prevista)."""

import numpy as np

from src.compare_goes_unsupervised_days import adaptive_threshold_day_grid, binary_metrics


def test_adaptive_raises_threshold_when_low_base_causes_many_fp():
    real = np.zeros((5, 5), dtype=np.float32)
    real[2, 2] = 1.0
    pred = np.full((5, 5), 0.6, dtype=np.float32)
    pred[2, 2] = 0.95
    thr, meta = adaptive_threshold_day_grid(real, pred, base_thr=0.2, min_recall=0.5, min_precision_floor=0.05)
    assert thr >= 0.2
    m = binary_metrics(real, (pred >= thr).astype(np.float32))
    assert m["fp"] <= binary_metrics(real, (pred >= 0.2).astype(np.float32))["fp"]


def test_adaptive_respects_base_floor():
    real = np.zeros((4, 4), dtype=np.float32)
    real[0, 0] = 1.0
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[0, 0] = 0.9
    pred[3, 3] = 0.95
    thr, _ = adaptive_threshold_day_grid(real, pred, base_thr=0.5)
    assert thr >= 0.5
