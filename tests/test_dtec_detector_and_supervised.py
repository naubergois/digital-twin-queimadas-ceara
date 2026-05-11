"""Testes dos módulos DTEC (detector e cabeça supervisionada)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from src.dtec_detector import DTECConfig, detect_dtec
from src.dtec_supervised import (
    DTECSupervisedConfig,
    blockwise_spatial_folds,
    build_features,
    stack_features,
    train_dtec_supervised,
)


def _synthetic_fire_scene(shape=(40, 40), n_hours=3, *, seed=0) -> tuple:
    """
    Cena com focos pontuais persistentes (pixels isolados de fogo) e
    fundo homogéneo com ruído moderado e ``BTD`` ligeiramente alto fora
    dos focos (simulando glint difuso, não fogo).

    Pontos isolados em vez de patch maciço — assim o detector multi-escala
    (residual local) tem residual elevado nos focos.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    truth = np.zeros(shape, dtype=bool)
    pts = [(10, 12), (10, 28), (20, 20), (28, 14), (28, 30)]
    for r, c in pts:
        truth[r, c] = True

    hourly: List[Dict[int, np.ndarray]] = []
    for _ in range(n_hours):
        bt13 = rng.normal(300.0, 0.5, size=shape).astype(np.float64)
        bt13[truth] += rng.normal(9.0, 0.5, size=int(truth.sum()))
        bt14 = bt13 - 3.5
        btd_extra = np.where(
            truth,
            rng.normal(13.5, 0.4, size=shape),
            rng.normal(14.0, 0.8, size=shape),
        )
        bt7 = bt14 + btd_extra
        hourly.append({7: bt7, 13: bt13, 14: bt14})
    valid = np.ones(shape, dtype=bool)
    return hourly, truth, valid


def test_detect_dtec_returns_shape_and_risk():
    hourly, truth, valid = _synthetic_fire_scene()
    cfg = DTECConfig(risk_top_percentile=92.0, min_active_hours=2)
    pred, risk = detect_dtec(hourly, valid, cfg=cfg)
    assert pred.shape == truth.shape
    assert risk.shape == truth.shape
    assert pred.dtype == bool
    assert np.isfinite(risk).all()


def test_detect_dtec_synthetic_recovers_focos():
    """
    Cena sintética com patch quente persistente em BTD moderado. Relaxa os
    percentis ``btd_*`` e ``bt7_glint`` porque o cenário não tem reflexão
    solar real — a configuração defensiva por defeito é desnecessária aqui.
    """
    hourly, truth, valid = _synthetic_fire_scene(seed=1)
    cfg = DTECConfig(
        risk_top_percentile=92.0,
        min_active_hours=2,
        btd_low_percentile=10.0,
        btd_high_percentile=99.5,
        bt7_glint_percentile=100.0,
        max_component_cells=0,
    )
    pred, _ = detect_dtec(hourly, valid, cfg=cfg)
    overlap = float(np.sum(pred & truth))
    # Pelo menos alguma sobreposição: signal sintético > ruído
    assert overlap > 0, f"sem sobreposição: pred={int(pred.sum())} truth={int(truth.sum())}"


def test_build_features_keys():
    hourly, _, valid = _synthetic_fire_scene()
    feats = build_features(hourly, valid)
    expected = {"bt13_max", "bt7_max", "btd_median", "twin_risk", "bt13_anom_21", "persist_h"}
    assert expected.issubset(feats.keys())
    for v in feats.values():
        assert v.shape == valid.shape


def test_stack_features_round_trip():
    hourly, _, valid = _synthetic_fire_scene()
    feats = build_features(hourly, valid)
    X, shape = stack_features(feats)
    assert X.shape[0] == valid.size
    assert shape == valid.shape


def test_train_dtec_supervised_logreg_predicts_in_range():
    hourly, truth, valid = _synthetic_fire_scene(seed=2)
    feats = build_features(hourly, valid)
    cfg = DTECSupervisedConfig(classifier="logreg")
    model = train_dtec_supervised(feats, truth, valid, cfg=cfg)
    proba = model.predict_proba_grid(feats, valid)
    assert proba.shape == valid.shape
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_train_dtec_supervised_hgb_runs():
    pytest.importorskip("sklearn.ensemble", reason="HGB precisa de sklearn moderno")
    hourly, truth, valid = _synthetic_fire_scene(seed=3)
    feats = build_features(hourly, valid)
    cfg = DTECSupervisedConfig(classifier="hgb")
    model = train_dtec_supervised(feats, truth, valid, cfg=cfg)
    proba = model.predict_proba_grid(feats, valid)
    assert proba.shape == valid.shape


def test_nms_peaks_subset_of_threshold_mask():
    hourly, truth, valid = _synthetic_fire_scene(seed=4)
    feats = build_features(hourly, valid)
    cfg = DTECSupervisedConfig(classifier="hgb")
    model = train_dtec_supervised(feats, truth, valid, cfg=cfg)
    # Picos com threshold T devem ser subconjunto do mask P>=T
    thr = 0.5
    mask = model.predict_mask(feats, valid, threshold=thr)
    peaks = model.predict_nms(feats, valid, threshold=thr, nms_radius=2)
    assert peaks.shape == mask.shape
    # Todos os picos devem estar dentro do mask
    assert np.all(~peaks | mask)


def test_blockwise_spatial_folds_partition():
    folds = blockwise_spatial_folds((30, 30), n_blocks_lat=3, n_blocks_lon=2)
    assert len(folds) == 6
    union = np.zeros((30, 30), dtype=bool)
    for train_mask, test_mask in folds:
        assert train_mask.shape == (30, 30)
        assert test_mask.shape == (30, 30)
        # train e test cobrem a cena toda e não se sobrepõem
        assert np.array_equal(train_mask | test_mask, np.ones((30, 30), dtype=bool))
        assert not (train_mask & test_mask).any()
        union |= test_mask
    # Os 6 blocos de teste em conjunto cobrem tudo
    assert union.all()
