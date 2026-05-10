"""Testes da comparação real vs previsto (GOES não supervisionado)."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.compare_goes_unsupervised_days import (
    compare_fire_days_and_save_figures,
    daily_pred_grid,
    daily_real_grid,
    binary_metrics,
)
from src.goes_unsupervised_twin import GOESUnsupervisedConfig
def test_daily_grids_and_metrics():
    n_lat, n_lon, T = 8, 8, 20
    times = pd.date_range("2024-08-01", periods=T, freq="h")
    rng = np.random.default_rng(0)
    cube = np.clip((0.1 * (1 + rng.standard_normal((T, n_lat, n_lon)))).astype(np.float32) ** 2, 0, 1)
    day0 = times[0].date()
    pred, n_fr = daily_pred_grid(cube, times, day0)
    assert pred.shape == (n_lat, n_lon)
    assert n_fr == T

    df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-08-01 12:00")] * 3,
            "lat": [-5.2, -5.3, -5.25],
            "lon": [-39.0, -39.1, -39.05],
        }
    )
    df["date"] = df["datetime"].dt.date
    real = daily_real_grid(df, day0, n_lat, n_lon)
    assert float(real.sum()) >= 1.0
    m = binary_metrics(real, (pred > 0.3).astype(np.float32))
    assert "iou" in m and m["tp"] >= 0


def test_compare_fire_days_writes_png_and_json():
    rows = []
    for d in range(4):
        for k in range(8):
            rows.append(
                {
                    "datetime": pd.Timestamp("2024-09-15") + pd.Timedelta(days=d, hours=k),
                    "lat": -5.4 + 0.01 * k,
                    "lon": -39.2 + 0.01 * k,
                }
            )
    df = pd.DataFrame(rows)
    cfg = GOESUnsupervisedConfig(grid_resolution=1.0, max_days_history=10, frame_minutes=120)
    with tempfile.TemporaryDirectory() as tmp:
        summary = compare_fire_days_and_save_figures(
            df,
            output_dir=tmp,
            cfg=cfg,
            netcdf_path=None,
            max_days=3,
            pred_threshold=0.2,
        )
        assert summary["days_evaluated"] >= 1
        p = Path(tmp)
        assert (p / "metrics_by_day.json").is_file()
        pngs = list(p.glob("goes_unsup_real_vs_pred_*.png"))
        assert len(pngs) >= 1
