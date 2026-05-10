"""Testes do preditor GOES-16 não supervisionado."""

import numpy as np
import pandas as pd

from src.goes_unsupervised_twin import (
    GOESUnsupervisedConfig,
    predict_unsupervised_fire_risk,
    resample_risk_to_shape,
    run_goes16_unsupervised_from_foci,
)


def test_predict_unsupervised_risk_shape_and_range():
    T, H, W = 12, 10, 12
    rng = np.random.default_rng(0)
    bt7 = rng.normal(48.0, 1.0, (T, H, W)).astype(np.float32)
    bt14 = rng.normal(42.0, 0.8, (T, H, W)).astype(np.float32)
    delta = (bt7 - bt14).astype(np.float32)
    res = rng.normal(0, 0.3, (T, H, W)).astype(np.float32)
    risk = predict_unsupervised_fire_risk(bt7, bt14, delta, res, GOESUnsupervisedConfig())
    assert risk.shape == (H, W)
    assert float(risk.min()) >= 0.0 and float(risk.max()) <= 1.0


def test_resample_risk_to_shape():
    r = np.ones((4, 5), dtype=np.float32) * 0.5
    out = resample_risk_to_shape(r, 8, 10)
    assert out.shape == (8, 10)


def test_run_goes16_unsupervised_from_foci_smoke():
    rows = []
    base = pd.Timestamp("2024-07-01")
    for d in range(5):
        for _ in range(4):
            rows.append(
                {
                    "datetime": base + pd.Timedelta(days=d, hours=3),
                    "lat": -5.2,
                    "lon": -39.1,
                    "municipio": "TEST",
                }
            )
    df = pd.DataFrame(rows)
    cfg = GOESUnsupervisedConfig(grid_resolution=1.0, max_days_history=10, frame_minutes=60)
    rep = run_goes16_unsupervised_from_foci(df, cfg=cfg)
    assert "_risk_grid" in rep
    assert rep["max_risk"] >= 0.0
    assert isinstance(rep["top_peaks"], list)
