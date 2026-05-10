"""Testes comparação ST-HyperNet por dia."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from src.compare_st_hypernet_days import compare_st_hypernet_fire_days_and_save_figures
from src.st_hypernet import STHyperNetConfig


def test_compare_st_hypernet_writes_outputs():
    rows = []
    for d in range(3):
        for h in range(6):
            rows.append(
                {
                    "datetime": pd.Timestamp("2024-10-01") + pd.Timedelta(days=d, hours=h),
                    "lat": -5.3 + 0.02 * h,
                    "lon": -39.0 + 0.02 * h,
                }
            )
    df = pd.DataFrame(rows)
    cfg = STHyperNetConfig(
        grid_resolution=1.5,
        frame_minutes=120,
        max_days_history=8,
        epochs=1,
        max_patches_per_epoch=96,
        batch_size=32,
        infer_batch=128,
        inference_stride=2,
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary = compare_st_hypernet_fire_days_and_save_figures(
            df,
            output_dir=tmp,
            cfg=cfg,
            max_days=2,
            pred_threshold=0.25,
        )
        assert summary["days_evaluated"] >= 1
        p = Path(tmp)
        assert (p / "metrics_by_day.json").is_file()
        assert list(p.glob("st_hypernet_real_vs_pred_*.png"))
        js = json.loads((p / "metrics_by_day.json").read_text(encoding="utf-8"))
        row0 = js["metrics_by_day"][0]
        assert "map_real_latlon" in row0 and isinstance(row0["map_real_latlon"], list)
        assert "map_pred_latlon_score" in row0 and isinstance(row0["map_pred_latlon_score"], list)
