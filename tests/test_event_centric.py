"""Testes da avaliação event-centric (``src/event_centric.py``)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.event_centric import (
    _build_grid_centers,
    _haversine_like_km,
    day_window_utc,
    evaluate_event_centric,
    evaluate_event_centric_multi_radius,
)


BBOX = {"min_lat": -8.0, "max_lat": -2.0, "min_lon": -42.0, "max_lon": -36.0}
GRID = (60, 60)


def _df(latlon):
    return pd.DataFrame(
        {
            "lat": [p[0] for p in latlon],
            "lon": [p[1] for p in latlon],
            "datetime": pd.to_datetime(
                ["2024-10-31T16:48:00Z"] * len(latlon), utc=True
            ),
        }
    )


def test_haversine_like_km_zero_diagonal():
    a = np.array([-5.0])
    b = np.array([-39.0])
    D = _haversine_like_km(a, b, a, b)
    assert D.shape == (1, 1)
    assert float(D[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_haversine_like_km_one_degree_lat():
    a = np.array([-5.0])
    b = np.array([-39.0])
    a2 = np.array([-4.0])
    D = _haversine_like_km(a, b, a2, b)
    assert float(D[0, 0]) == pytest.approx(111.32, rel=1e-2)


def test_grid_centers_shape_and_bbox():
    lat_g, lon_g = _build_grid_centers(BBOX, GRID)
    assert lat_g.shape == GRID
    assert lon_g.shape == GRID
    assert lat_g.min() > BBOX["min_lat"]
    assert lat_g.max() < BBOX["max_lat"]
    assert lon_g.min() > BBOX["min_lon"]
    assert lon_g.max() < BBOX["max_lon"]


def test_event_centric_perfect_match_cell_mode():
    pred = np.zeros(GRID, dtype=bool)
    # Célula central
    pred[30, 30] = True
    lat_g, lon_g = _build_grid_centers(BBOX, GRID)
    df = _df([(float(lat_g[30, 30]), float(lon_g[30, 30]))])

    m = evaluate_event_centric(
        pred, df, BBOX, GRID, radius_km=2.0, matching="cell"
    )
    assert m.f1 == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.tp_for_recall == 1
    assert m.tp_for_precision == 1


def test_event_centric_far_no_match():
    pred = np.zeros(GRID, dtype=bool)
    pred[5, 5] = True
    lat_g, lon_g = _build_grid_centers(BBOX, GRID)
    # Foco a > 100 km do pixel previsto
    df = _df([(float(lat_g[55, 55]), float(lon_g[55, 55]))])

    m = evaluate_event_centric(
        pred, df, BBOX, GRID, radius_km=10.0, matching="cell"
    )
    assert m.f1 == pytest.approx(0.0)
    assert m.fn_focos == 1
    assert m.fp_components == 1


def test_event_centric_centroid_vs_cell_modes():
    """Blob grande: centróide pode estar longe; modo cell ainda casa."""
    pred = np.zeros(GRID, dtype=bool)
    pred[20:25, 20:25] = True  # blob 5x5
    lat_g, lon_g = _build_grid_centers(BBOX, GRID)
    # Foco perto da borda do blob
    df = _df([(float(lat_g[20, 20]), float(lon_g[20, 20]))])

    m_cell = evaluate_event_centric(
        pred, df, BBOX, GRID, radius_km=2.0, matching="cell"
    )
    m_cent = evaluate_event_centric(
        pred, df, BBOX, GRID, radius_km=2.0, matching="centroid"
    )
    assert m_cell.tp_for_recall == 1
    # centroid pode falhar se distância > 2 km
    assert m_cent.recall <= m_cell.recall


def test_event_centric_no_focos_or_no_components():
    empty = np.zeros(GRID, dtype=bool)
    df_empty = _df([])
    m = evaluate_event_centric(empty, df_empty, BBOX, GRID, radius_km=3.0)
    assert m.f1 == 0.0 and m.precision == 0.0 and m.recall == 0.0
    assert m.n_focos == 0 and m.n_components == 0


def test_event_centric_multi_radius_monotonic():
    """Aumentar R nunca reduz nº de TP_recall (focos cobertos)."""
    pred = np.zeros(GRID, dtype=bool)
    pred[30, 30] = True
    lat_g, lon_g = _build_grid_centers(BBOX, GRID)
    df = _df([(float(lat_g[30, 30] + 0.05), float(lon_g[30, 30]))])  # ~5 km Norte

    out = evaluate_event_centric_multi_radius(
        pred,
        df,
        BBOX,
        GRID,
        radii_km=(1.0, 3.0, 6.0, 12.0),
    )
    tp_seq = [out[k]["ec_tp_recall"] for k in ("R=1.0km", "R=3.0km", "R=6.0km", "R=12.0km")]
    assert tp_seq == sorted(tp_seq), tp_seq


def test_day_window_utc_basic():
    d0, d1 = day_window_utc("2024-10-31")
    assert d0 == datetime(2024, 10, 31, tzinfo=timezone.utc)
    assert (d1 - d0).total_seconds() == 86400.0


def test_event_centric_valid_bins_filters_pred():
    pred = np.ones(GRID, dtype=bool)
    valid = np.zeros(GRID, dtype=bool)
    valid[:5, :5] = True
    df = _df([])
    m = evaluate_event_centric(pred, df, BBOX, GRID, valid_bins=valid)
    assert m.n_components <= int(valid.sum())
