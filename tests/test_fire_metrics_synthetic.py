"""
Benchmark sintético alinhado (verdade na mesma grade que o detector).

.. warning::

    Comparar GOES não supervisionado com INPE em grade grossa **não** deve esperar
    F1/precisão ≳ 0,8 em dados reais sem produtos AF ou métricas por evento.
    Aqui o “fogo” é um patch térmico persistente + BTD artificial coerente;
    o teste garante que **combined_persistence**, o **gêmeo digital** e a **fusão AND**
    conseguem atingir limiares altos quando o problema está bem posto.

Objetivo: ``precision > 0.8`` e ``f1 > 0.8`` para cada modo (com grelha de ``contamination``).
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pytest

from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig
from src.goes_fire_method_v2 import (
    CombinedPersistenceConfig,
    detect_combined_persistence,
    fuse_masks_intersection,
)

TARGET_P = 0.80
TARGET_F1 = 0.80


def _prf(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> Tuple[float, float, float]:
    p = pred & valid
    t = truth & valid
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _find_contamination(
    pred_fn: Callable[[float], np.ndarray],
    truth: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Procura ``c`` tal que precisión e F1 ≥ alvos (varredura não supervisionada típica)."""
    best_tuple: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)
    for c in np.linspace(0.006, 0.24, 72):
        pred = pred_fn(float(c))
        prec, rec, f1 = _prf(pred, truth, valid)
        if prec >= TARGET_P and f1 >= TARGET_F1:
            return prec, rec, f1, float(c)
        # backup: guardar melhor compromisso min(p,f1)
        score = min(prec, f1)
        if score > min(best_tuple[0], best_tuple[2]):
            best_tuple = (prec, rec, f1, float(c))
    return best_tuple


def synthetic_persistent_fire(
    shape: Tuple[int, int] = (52, 52),
    n_hours: int = 4,
    *,
    seed: int = 42,
) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """Várias horas com patch quente + BTD elevado só no foco (persistência real)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    truth = np.zeros((h, w), dtype=bool)
    r0, r1 = h // 2 - 6, h // 2 + 6
    c0, c1 = w // 2 - 6, w // 2 + 6
    truth[r0:r1, c0:c1] = True

    hourly: List[dict] = []
    for _ in range(n_hours):
        bt13 = rng.normal(286.5, 0.35, size=(h, w)).astype(np.float64)
        bt13[truth] = rng.normal(311.0, 0.25, size=int(truth.sum())).astype(np.float64)
        bt14 = bt13 - 4.2
        bt7 = bt13 + np.where(truth, rng.normal(22.0, 0.5, size=(h, w)), rng.normal(4.0, 0.4, size=(h, w)))
        hourly.append({7: bt7.astype(np.float64), 13: bt13, 14: bt14.astype(np.float64)})

    valid = np.ones((h, w), dtype=bool)
    return hourly, truth, valid


@pytest.fixture
def synth_bundle():
    return synthetic_persistent_fire()


@pytest.fixture
def cp_relaxed_cfg() -> CombinedPersistenceConfig:
    """Configuração permissiva no benchmark sintético (sem ruído espacial extra)."""
    return CombinedPersistenceConfig(
        hour_active_percentile=72.0,
        adaptive_hour_percentile=False,
        weights_peak_mean_persist=(0.45, 0.25, 0.30),
        min_persist_frac=0.0,
        persist_floor=0.0,
        persist_scale=0.0,
        min_active_hours=0,
        weak_open_after_pred_cells=0,
        weak_open_iterations=0,
        min_component_cells=1,
        morph_open_iterations=0,
        robust_norm_quantiles=(4.0, 96.0),
    )


@pytest.fixture
def twin_relaxed_cfg() -> GOESFireDigitalTwinConfig:
    """Gêmeo sem LOF (pequena grade) + suavização moderada para estabilizar percentis."""
    return GOESFireDigitalTwinConfig(
        persistence=0.58,
        gaussian_sigma=1.0,
        dbt_weight=0.52,
        fusion="prob_or",
        threshold_mode="percentile",
        lof_neighbors=0,
        multiscale_median_sizes=(5, 9),
    )


def test_combined_persistence_synthetic_high_precision_and_f1(synth_bundle, cp_relaxed_cfg):
    hourly, truth, valid = synth_bundle

    def pred_fn(c: float) -> np.ndarray:
        return detect_combined_persistence(hourly, valid, c, cfg=cp_relaxed_cfg)

    prec, rec, f1, c_hit = _find_contamination(pred_fn, truth, valid)
    assert prec > TARGET_P and f1 > TARGET_F1, (
        f"combined_persistence: prec={prec:.3f} f1={f1:.3f} rec={rec:.3f} "
        f"(melhor c tentado≈{c_hit}); subir contraste sintético ou alargar patch."
    )


def test_digital_twin_synthetic_high_precision_and_f1(synth_bundle, twin_relaxed_cfg):
    hourly, truth, valid = synth_bundle
    h, w = truth.shape
    twin = GOESFireDigitalTwin((h, w), twin_relaxed_cfg)
    twin.ingest_series(hourly)

    def pred_fn(c: float) -> np.ndarray:
        return twin.predict_mask(c, valid)

    prec, rec, f1, c_hit = _find_contamination(pred_fn, truth, valid)
    assert prec > TARGET_P and f1 > TARGET_F1, (
        f"digital_twin: prec={prec:.3f} f1={f1:.3f} rec={rec:.3f} (c≈{c_hit})"
    )


def test_fused_twin_and_combined_intersection_high_precision_and_f1(
    synth_bundle,
    cp_relaxed_cfg,
    twin_relaxed_cfg,
):
    """
    Técnica extra: **intersecção** twin ∩ combined_persistence — reduz FP quando ambos concordam.
    """
    hourly, truth, valid = synth_bundle
    h, w = truth.shape
    twin = GOESFireDigitalTwin((h, w), twin_relaxed_cfg)
    twin.ingest_series(hourly)

    def pred_fn(c: float) -> np.ndarray:
        p_t = twin.predict_mask(c, valid)
        p_c = detect_combined_persistence(hourly, valid, c, cfg=cp_relaxed_cfg)
        return fuse_masks_intersection(p_t, p_c)

    prec, rec, f1, c_hit = _find_contamination(pred_fn, truth, valid)
    assert prec > TARGET_P and f1 > TARGET_F1, (
        f"fusão AND: prec={prec:.3f} f1={f1:.3f} rec={rec:.3f} (c≈{c_hit})"
    )
