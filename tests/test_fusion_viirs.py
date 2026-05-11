"""Testes da fusão multi-sensor GOES + VIIRS (`src/multi_sensor_fusion.py`)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.firms_download import FIRMSRequest, firms_url, offline_demo_viirs
from src.multi_sensor_fusion import (
    FusionConfig,
    fuse_goes_with_viirs,
    load_viirs_firms_csv,
    synthesize_viirs_proxy,
    viirs_cell_mask,
)


BBOX = {"min_lat": -8.0, "max_lat": -2.0, "min_lon": -42.0, "max_lon": -36.0}
GRID = (60, 60)


def _truth_focos() -> pd.DataFrame:
    return pd.DataFrame({
        "lat": [-5.0, -5.2, -4.8, -6.0, -3.5],
        "lon": [-39.0, -39.1, -38.9, -40.0, -38.5],
        "datetime": pd.to_datetime(
            ["2024-10-31T16:48:00Z"] * 5, utc=True
        ),
    })


def test_synthesize_viirs_proxy_obeys_detection_rate():
    df_truth = _truth_focos()
    np.random.seed(0)
    out = synthesize_viirs_proxy(
        df_truth, bbox=BBOX,
        detection_rate=1.0,  # 100% → todos os focos detectados (com jitter)
        spatial_jitter_km=0.0,
        false_positive_rate=0.0,
        seed=0,
    )
    # detection_rate=1.0 e fp=0 → exatamente len(df_truth) linhas
    assert len(out) == len(df_truth)
    # Posições com jitter=0 devem ser idênticas (a menos da ordem)
    out_sorted = out.sort_values(["lat", "lon"]).reset_index(drop=True)
    t_sorted = df_truth.sort_values(["lat", "lon"]).reset_index(drop=True)
    np.testing.assert_allclose(out_sorted["lat"].to_numpy(), t_sorted["lat"].to_numpy(), atol=1e-9)


def test_synthesize_viirs_proxy_adds_false_positives():
    df_truth = _truth_focos()
    out = synthesize_viirs_proxy(
        df_truth, bbox=BBOX,
        detection_rate=0.0,
        false_positive_rate=0.1,
        n_cells_in_bbox=200,
        seed=1,
    )
    # detection_rate=0 → só falsos positivos
    assert len(out) == int(0.1 * 200)
    assert out["confidence"].iloc[0] == 60.0


def test_viirs_cell_mask_counts_cells():
    df_truth = _truth_focos()
    m = viirs_cell_mask(df_truth, BBOX, GRID)
    assert m.shape == GRID
    assert m.dtype == bool
    # Cada foco distinto deve cair numa célula → ≤ 5 cells (poderiam coincidir)
    assert 1 <= int(m.sum()) <= 5


def test_viirs_cell_mask_filters_confidence():
    df = pd.DataFrame({
        "lat": [-5.0, -5.0],
        "lon": [-39.0, -39.1],
        "datetime": pd.to_datetime(["2024-10-31T17:00:00Z"] * 2, utc=True),
        "confidence": [30.0, 90.0],
    })
    m_all = viirs_cell_mask(df, BBOX, GRID)
    m_high = viirs_cell_mask(df, BBOX, GRID, min_confidence=80.0)
    assert int(m_all.sum()) >= int(m_high.sum())


def test_fuse_goes_with_viirs_and_mode_intersects():
    truth = _truth_focos()
    goes_pred = np.zeros(GRID, dtype=bool)
    goes_pred[28:32, 28:32] = True  # bloco no centro
    valid = np.ones(GRID, dtype=bool)
    res = fuse_goes_with_viirs(
        goes_pred, None, truth, BBOX, GRID, valid,
        cfg=FusionConfig(mode="and", gate_radius_km=10.0),
    )
    # AND nunca produz mais cells do que goes_pred
    assert int(res.pred_mask.sum()) <= int(goes_pred.sum())


def test_fuse_goes_with_viirs_or_mode_unions():
    truth = _truth_focos()
    goes_pred = np.zeros(GRID, dtype=bool)
    goes_pred[5, 5] = True
    valid = np.ones(GRID, dtype=bool)
    res = fuse_goes_with_viirs(
        goes_pred, None, truth, BBOX, GRID, valid,
        cfg=FusionConfig(mode="or", gate_radius_km=5.0),
    )
    assert int(res.pred_mask.sum()) >= int(goes_pred.sum())


def test_fuse_weighted_requires_prob():
    truth = _truth_focos()
    valid = np.ones(GRID, dtype=bool)
    with pytest.raises(ValueError, match="goes_prob"):
        fuse_goes_with_viirs(
            np.zeros(GRID, dtype=bool), None, truth, BBOX, GRID, valid,
            cfg=FusionConfig(mode="weighted"),
        )


def test_fuse_weighted_runs_with_prob():
    truth = _truth_focos()
    goes_pred = np.zeros(GRID, dtype=bool)
    goes_pred[28:32, 28:32] = True
    prob = np.zeros(GRID, dtype=np.float64)
    prob[28:32, 28:32] = 0.85
    valid = np.ones(GRID, dtype=bool)
    res = fuse_goes_with_viirs(
        goes_pred, prob, truth, BBOX, GRID, valid,
        cfg=FusionConfig(mode="weighted", gate_radius_km=5.0,
                         weight_goes=0.6, weight_viirs=0.4),
    )
    assert res.mode == "weighted"
    assert res.pred_mask.shape == GRID


def test_load_viirs_firms_csv_normalizes_columns(tmp_path: Path):
    csv = (
        "latitude,longitude,brightness,acq_date,acq_time,confidence,frp\n"
        "-5.10,-39.20,330.5,2024-10-31,1648,h,12.3\n"
        "-4.95,-39.05,310.1,2024-10-31,1648,n,8.1\n"
    )
    p = tmp_path / "viirs.csv"
    p.write_text(csv)
    df = load_viirs_firms_csv(p)
    assert "lat" in df.columns and "lon" in df.columns
    assert "datetime" in df.columns
    assert df["confidence"].iloc[0] == 90.0  # h → 90
    assert df["datetime"].iloc[0].hour == 16


def test_firms_url_includes_components():
    req = FIRMSRequest(
        source="VIIRS_NOAA20_NRT",
        bbox=(-41.5, -7.9, -37.0, -2.5),
        day=datetime(2024, 10, 31, tzinfo=timezone.utc).date(),
        range_days=1,
    )
    url = firms_url(req, "MY_KEY")
    assert "VIIRS_NOAA20_NRT" in url
    assert "MY_KEY" in url
    assert "2024-10-31" in url
    assert "-41.5,-7.9,-37.0,-2.5" in url


def test_offline_demo_viirs_returns_df():
    truth = _truth_focos()
    out = offline_demo_viirs(truth, BBOX, detection_rate=0.5, seed=99)
    assert isinstance(out, pd.DataFrame)
    assert "lat" in out.columns and "lon" in out.columns
