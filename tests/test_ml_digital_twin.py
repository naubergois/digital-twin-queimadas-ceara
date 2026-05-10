"""Testes do gêmeo ML + propagação — regressão, calibração e falsos positivos."""

import numpy as np
import pandas as pd
import pytest

from src.ml_digital_twin import FireMLDigitalTwin, MLTwinConfig


def _tiny_twin(cfg: MLTwinConfig | None = None) -> FireMLDigitalTwin:
    c = cfg or MLTwinConfig(grid_resolution=2.0, lookback_days=2)
    return FireMLDigitalTwin(c)


def test_lat_lon_to_grid_clamps_inside_bbox():
    twin = _tiny_twin()
    i, j = twin._lat_lon_to_grid(-7.9, -41.5)
    assert 0 <= i < twin.n_lat and 0 <= j < twin.n_lon
    i2, j2 = twin._lat_lon_to_grid(-2.5, -37.0)
    assert i2 == twin.n_lat - 1 and j2 == twin.n_lon - 1


def test_prepare_daily_grids_binary_presence():
    twin = _tiny_twin()
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-06-01 12:00", "2024-06-01 14:00"]),
            "lat": [-5.0, -5.1],
            "lon": [-39.0, -39.0],
            "municipio": ["X", "X"],
        }
    )
    dates, grids, dfx = twin._prepare_daily_grids(df)
    assert len(dates) >= 1
    assert grids.shape[1:] == (twin.n_lat, twin.n_lon)
    assert float(grids.max()) <= 1.0
    assert (grids[0] >= 0).all()


def test_simulate_twin_zero_proba_no_fire():
    twin = _tiny_twin(
        MLTwinConfig(grid_resolution=2.0, twin_pday_weight=1.0, twin_cooldown_days=1)
    )
    d, lat, lon = 3, twin.n_lat, twin.n_lon
    p = np.zeros((d, lat, lon), dtype=np.float32)
    out = twin._simulate_twin(p, twin_spread_threshold=0.5)
    assert out.shape == (d, lat, lon)
    assert int(out.sum()) == 0


def test_simulate_twin_high_threshold_reduces_spread_vs_low():
    """Vizinhos só ‘pegam fogo’ com limiar baixo — regressão para modo low_fp."""
    twin = _tiny_twin(
        MLTwinConfig(
            grid_resolution=0.5,
            twin_pday_weight=0.9,
            twin_cooldown_days=1,
        )
    )
    lat, lon = twin.n_lat, twin.n_lon
    ci, cj = lat // 2, lon // 2
    p = np.full((8, lat, lon), 0.08, dtype=np.float32)
    p[:, ci, cj] = 0.95
    low = twin._simulate_twin(p, twin_spread_threshold=0.12)
    high = twin._simulate_twin(p, twin_spread_threshold=0.55)
    assert low.sum() >= high.sum()


def test_spatial_tolerant_scores_perfect_match():
    twin = _tiny_twin()
    g = np.zeros((2, twin.n_lat, twin.n_lon), dtype=np.float32)
    g[:, 1, 1] = 1.0
    p = g.copy()
    s = twin._spatial_tolerant_scores(g, p, radius=0)
    assert s["precision_tolerant"] == pytest.approx(1.0)
    assert s["recall_tolerant"] == pytest.approx(1.0)


def test_calibrate_low_fp_prefers_higher_proba_when_noise_floor_exists():
    """Negativos com p abaixo do positivo: low_fp não deve escolher limiar abaixo do f1 'seguro'."""
    twin_f1 = FireMLDigitalTwin(
        MLTwinConfig(
            grid_resolution=2.0,
            min_recall_target=0.6,
            min_precision_target=0.05,
            calibration_objective="f1",
            auto_calibrate=True,
        )
    )
    twin_fp = FireMLDigitalTwin(
        MLTwinConfig(
            grid_resolution=2.0,
            min_recall_target=0.6,
            min_precision_target=0.2,
            calibration_objective="low_fp",
            auto_calibrate=True,
        )
    )
    lat, lon = twin_f1.n_lat, twin_f1.n_lon
    n_days = 4
    y = np.zeros((n_days, lat, lon), dtype=np.float32)
    pr = np.zeros((n_days, lat, lon), dtype=np.float32)
    for d in range(n_days):
        y[d, min(2, lat - 1), min(2, lon - 1)] = 1.0
        pr[d, :, :] = 0.41
        pr[d, min(2, lat - 1), min(2, lon - 1)] = 0.88
    pth_f1, _, rep_f1 = twin_f1._calibrate_thresholds(y, pr)
    pth_fp, _, rep_fp = twin_fp._calibrate_thresholds(y, pr)
    assert rep_f1["chosen_proba"]["precision"] <= rep_fp["chosen_proba"]["precision"] + 1e-6
    assert pth_fp >= 0.4
    assert "twin_precision" in rep_fp["chosen_twin"]


@pytest.mark.slow
def test_validate_with_real_data_runs_on_synthetic_series():
    """Treino completo (lento): usa grade grosseira e poucos modelos candidatos."""
    np.random.default_rng(42)

    cfg = MLTwinConfig(
        grid_resolution=1.0,
        lookback_days=3,
        test_ratio=0.25,
        auto_calibrate=True,
        benchmark_sample_size=8000,
        min_recall_target=0.35,
        min_precision_target=0.05,
        calibration_objective="low_fp",
        tolerant_radius_cells=2,
        max_positive_rate=0.35,
        use_hard_negative_mining=False,
    )
    twin = FireMLDigitalTwin(cfg)
    twin.model_candidates = {
        "hist_gb": twin.model_candidates["hist_gb"],
    }

    rows = []
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2023-05-01")
    for day in range(40):
        dt = base + pd.Timedelta(days=day)
        n = 6 if 10 <= day <= 30 else 1
        for _ in range(n):
            rows.append(
                {
                    "datetime": dt + pd.Timedelta(hours=int(rng.integers(0, 20))),
                    "lat": float(rng.uniform(-6.5, -5.5)),
                    "lon": float(rng.uniform(-40.0, -38.5)),
                    "municipio": "TESTOPOLIS",
                }
            )
    df = pd.DataFrame(rows)
    out = twin.validate_with_real_data(df)
    assert "ml_metrics" in out and "twin_metrics" in out
    assert out["ml_metrics"]["precision"] >= 0.0
    assert out["config"]["calibration_objective"] == "low_fp"
