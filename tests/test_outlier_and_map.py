"""Testes da camada de outliers (`src/dtec_outlier.py`) e do mapa (`src/map_view.py`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dtec_outlier import (
    OutlierConfig,
    filter_predictions_by_outlier,
    outlier_mask_from_twin_features,
)
from src.map_view import build_map, build_multi_date_index, save_map


BBOX = {"min_lat": -8.0, "max_lat": -2.0, "min_lon": -42.0, "max_lon": -36.0}
GRID = (40, 40)


def _synthetic_features(seed=0):
    rng = np.random.default_rng(seed)
    base_bt13 = rng.normal(300.0, 0.5, GRID).astype(np.float64)
    base_bt7 = rng.normal(315.0, 1.0, GRID).astype(np.float64)
    base_btd = rng.normal(13.5, 0.6, GRID).astype(np.float64)
    twin_risk = rng.uniform(0.0, 0.3, GRID).astype(np.float64)
    bt13_anom = rng.normal(0.0, 0.3, GRID).astype(np.float64)
    persist = rng.choice([0, 1, 2], size=GRID, p=[0.6, 0.3, 0.1]).astype(np.float64)

    # 5 píxeis "outlier" claros (fogos sintéticos): twin_risk alto + BT13 alto + persist alto
    outlier_pts = [(8, 10), (12, 25), (20, 20), (28, 30), (35, 14)]
    for r, c in outlier_pts:
        base_bt13[r, c] = 312.0
        base_bt7[r, c] = 325.0
        base_btd[r, c] = 14.5
        twin_risk[r, c] = 0.85
        bt13_anom[r, c] = 4.0
        persist[r, c] = 3

    feats = {
        "twin_risk": twin_risk,
        "bt13_max": base_bt13,
        "bt7_max": base_bt7,
        "btd_median": base_btd,
        "bt13_anom_21": bt13_anom,
        "persist_h": persist,
    }
    valid = np.ones(GRID, dtype=bool)
    truth = np.zeros(GRID, dtype=bool)
    for r, c in outlier_pts:
        truth[r, c] = True
    return feats, valid, truth


@pytest.mark.parametrize("method", ["isolation_forest", "local_outlier_factor", "elliptic_envelope", "ensemble"])
def test_outlier_mask_shape_and_dtype(method):
    feats, valid, _ = _synthetic_features(seed=1)
    cfg = OutlierConfig(method=method, contamination=0.02)
    m = outlier_mask_from_twin_features(feats, valid, cfg=cfg)
    assert m.shape == valid.shape
    assert m.dtype == bool


def test_outlier_recovers_synthetic_outliers():
    """Outliers sintéticos com twin_risk alto têm de ser apanhados por IF."""
    feats, valid, truth = _synthetic_features(seed=2)
    cfg = OutlierConfig(method="isolation_forest", contamination=0.02)
    m = outlier_mask_from_twin_features(feats, valid, cfg=cfg)
    overlap = float(np.sum(m & truth))
    # Esperamos pelo menos 3 dos 5 sintéticos
    assert overlap >= 3


def test_filter_predictions_by_outlier_is_subset_and():
    """Filtro AND: resultado é subconjunto da máscara de entrada."""
    feats, valid, _ = _synthetic_features(seed=3)
    candidate = np.zeros(GRID, dtype=bool)
    candidate[5:15, 5:15] = True
    cfg = OutlierConfig(method="isolation_forest", contamination=0.05)
    out = filter_predictions_by_outlier(candidate, feats, valid, cfg=cfg)
    assert out.shape == candidate.shape
    assert np.all(~out | candidate)


def test_outlier_returns_empty_when_too_few_samples():
    """Cenas < 50 cells não tentam ajustar o modelo: máscara fica vazia."""
    tiny = (5, 5)
    feats = {
        "twin_risk": np.zeros(tiny),
        "bt13_max": np.zeros(tiny),
        "bt7_max": np.zeros(tiny),
        "btd_median": np.zeros(tiny),
        "bt13_anom_21": np.zeros(tiny),
        "persist_h": np.zeros(tiny),
    }
    valid = np.ones(tiny, dtype=bool)
    m = outlier_mask_from_twin_features(feats, valid, cfg=OutlierConfig())
    assert m.shape == tiny
    assert not m.any()


def _focos_df():
    return pd.DataFrame(
        {
            "lat": [-5.0, -5.2, -4.8],
            "lon": [-39.0, -39.1, -38.9],
            "datetime": pd.to_datetime(
                ["2024-10-31T16:48:00Z"] * 3, utc=True
            ),
        }
    )


def test_build_map_writes_html(tmp_path: Path):
    pred = np.zeros(GRID, dtype=bool)
    pred[18:22, 18:22] = True
    out = tmp_path / "mapa.html"
    p = save_map(
        out,
        _focos_df(),
        pred,
        BBOX,
        GRID,
        day_iso="2024-10-31",
        radius_km=10.0,
        title="DTEC teste",
    )
    assert p.is_file()
    content = p.read_text(encoding="utf-8")
    assert "DTEC teste" in content
    assert "2024-10-31" in content
    # Folium gera div com leaflet
    assert "leaflet" in content.lower()


def test_build_map_with_no_predictions(tmp_path: Path):
    pred = np.zeros(GRID, dtype=bool)
    m = build_map(_focos_df(), pred, BBOX, GRID, day_iso="2024-10-31", radius_km=8.0)
    out = tmp_path / "vazio.html"
    m.save(str(out))
    txt = out.read_text(encoding="utf-8")
    assert "Focos INPE: <b>3</b>" in txt
    assert "Previstos: <b>0</b>" in txt


def test_build_multi_date_index(tmp_path: Path):
    p1 = tmp_path / "mapa_2024-10-31.html"
    p1.write_text("<html></html>")
    p2 = tmp_path / "mapa_2024-11-01.html"
    p2.write_text("<html></html>")
    idx = build_multi_date_index(
        [("2024-10-31", p1), ("2024-11-01", p2)],
        tmp_path / "index.html",
    )
    assert idx.is_file()
    html = idx.read_text(encoding="utf-8")
    assert "2024-10-31" in html and "2024-11-01" in html
    assert p1.name in html and p2.name in html
