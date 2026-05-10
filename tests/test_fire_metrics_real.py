"""
Testes com **dados reais** (GOES-16 CMIPF local + focos INPE).

Requisitos (senão ``pytest.skip``):

- ``data/inpe_focos_ce/focos_ce_INPE_2024_2026.csv`` (ou ajustar ``REAL_INPE_GLOB``)
- Para o dia fixo, NetCDF por canal em ``data/goes16_raw/`` com tag do ano-dia
  (com ``--skip-download`` o código usa **um** granulo por canal — várias horas repetem o mesmo ficheiro).

**Não** se exige F1/precisão altos: em dados reais isso depende de alinhamento semântico e temporal.
Os testes validam que o pipeline corre, produz máscaras coerentes e métricas finitas em [0, 1].
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from config.ceara_config import CEARA_BBOX
from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig
from src.goes_fire_method_v2 import (
    CombinedPersistenceConfig,
    detect_combined_persistence,
    fuse_masks_intersection,
)
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    evaluate_one_day_enhanced,
    find_local_goes_nc,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    merge_band_grids_max,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_RAW_DIR = REPO_ROOT / "data" / "goes16_raw"
REAL_INPE_CSV = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
REAL_DAY = date(2024, 10, 31)
REAL_HOURS = (16, 17, 18)
REAL_CHANNELS = (7, 13, 14)
GRID_HW: Tuple[int, int] = (48, 48)


def _require_local_assets() -> None:
    if not REAL_INPE_CSV.is_file():
        pytest.skip(f"CSV INPE ausente: {REAL_INPE_CSV}")
    if not REAL_RAW_DIR.is_dir():
        pytest.skip(f"Pasta GOES ausente: {REAL_RAW_DIR}")
    for ch in REAL_CHANNELS:
        try:
            find_local_goes_nc(REAL_RAW_DIR, REAL_DAY, ch)
        except FileNotFoundError:
            pytest.skip(f"NetCDF canal {ch} ausente para {REAL_DAY} em {REAL_RAW_DIR}")


def _assert_sane_metrics(m: Dict[str, Any], *, expect_truth: bool = True) -> None:
    for k in ("precision", "recall", "f1", "iou", "accuracy"):
        assert k in m
        assert np.isfinite(m[k])
        assert 0.0 <= float(m[k]) <= 1.0
    assert int(m["valid_cells"]) > 0
    assert int(m["pred_cells"]) >= 0
    if expect_truth:
        assert int(m["truth_cells"]) >= 0


@pytest.fixture(scope="module")
def real_hourly_and_df():
    _require_local_assets()
    df = load_inpe_focos(REAL_INPE_CSV)
    hourly, _ = collect_hourly_band_grids(
        REAL_DAY,
        REAL_HOURS,
        REAL_CHANNELS,
        CEARA_BBOX,
        GRID_HW,
        REAL_RAW_DIR,
        skip_download=True,
        overwrite=False,
        use_dqf=True,
        show_progress=False,
    )
    return hourly, df


@pytest.mark.real_data
def test_real_evaluate_combined_persistence_fixed_contamination(real_hourly_and_df):
    hourly, df = real_hourly_and_df
    band = merge_band_grids_max(hourly, REAL_CHANNELS)
    pairs = evaluate_one_day_enhanced(
        hourly,
        band,
        df,
        REAL_DAY,
        grid_hw=GRID_HW,
        contamination=0.06,
        method="combined_persistence",
        calibrate_contamination_flag=False,
        truth_dilate_iters=1,
        combined_persistence_cfg=CombinedPersistenceConfig(
            weak_open_after_pred_cells=0,
            min_component_cells=1,
            morph_open_iterations=0,
        ),
        calibrate_beta=1.0,
    )
    assert len(pairs) == 1
    res, pred = pairs[0]
    assert res.method == "combined_persistence"
    assert pred.shape == GRID_HW
    assert pred.dtype == bool
    _assert_sane_metrics(res.metrics)


@pytest.mark.real_data
def test_real_evaluate_digital_twin_fixed_contamination(real_hourly_and_df):
    hourly, df = real_hourly_and_df
    band = merge_band_grids_max(hourly, REAL_CHANNELS)
    twin_cfg = GOESFireDigitalTwinConfig(lof_neighbors=0, multiscale_median_sizes=(5, 9))
    pairs = evaluate_one_day_enhanced(
        hourly,
        band,
        df,
        REAL_DAY,
        grid_hw=GRID_HW,
        contamination=0.06,
        method="digital_twin",
        calibrate_contamination_flag=False,
        truth_dilate_iters=1,
        twin_cfg=twin_cfg,
        calibrate_beta=1.0,
    )
    assert len(pairs) == 1
    res, pred = pairs[0]
    assert pred.shape == GRID_HW
    _assert_sane_metrics(res.metrics)


@pytest.mark.real_data
@pytest.mark.slow
def test_real_combined_persistence_calibrated(real_hourly_and_df):
    """Calibra ``contamination`` ao INPE do dia (mais lento)."""
    hourly, df = real_hourly_and_df
    band = merge_band_grids_max(hourly, REAL_CHANNELS)
    pairs = evaluate_one_day_enhanced(
        hourly,
        band,
        df,
        REAL_DAY,
        grid_hw=GRID_HW,
        contamination=0.02,
        method="combined_persistence",
        calibrate_contamination_flag=True,
        truth_dilate_iters=1,
        combined_persistence_cfg=CombinedPersistenceConfig(
            weak_open_after_pred_cells=0,
            min_component_cells=1,
            morph_open_iterations=0,
        ),
        calibrate_beta=0.5,
    )
    res, pred = pairs[0]
    assert res.metrics["calibrated_contamination"] is True
    assert 0.004 <= res.contamination_used <= 0.5
    _assert_sane_metrics(res.metrics)


@pytest.mark.real_data
def test_real_fusion_twin_and_combined_intersection(real_hourly_and_df):
    """Fusão AND entre gêmeo digital e combined_persistence em dados reais (smoke)."""
    hourly, df = real_hourly_and_df
    band = merge_band_grids_max(hourly, REAL_CHANNELS)
    valid_bins = intersect_valid_bins_hourly(hourly, sorted(REAL_CHANNELS))
    h, w = GRID_HW
    twin = GOESFireDigitalTwin(
        (h, w),
        GOESFireDigitalTwinConfig(lof_neighbors=0, multiscale_median_sizes=(5, 9)),
    )
    twin.ingest_series(hourly)
    c = 0.06
    p_t = twin.predict_mask(c, valid_bins)
    cp_cfg = CombinedPersistenceConfig(
        weak_open_after_pred_cells=0,
        min_component_cells=1,
        morph_open_iterations=0,
    )
    p_c = detect_combined_persistence(hourly, valid_bins, c, cfg=cp_cfg)
    fused = fuse_masks_intersection(p_t, p_c)
    assert fused.shape == GRID_HW
    assert fused.dtype == bool
    assert int(np.sum(fused)) <= int(np.sum(p_t))
    assert int(np.sum(fused)) <= int(np.sum(p_c))

    pairs = evaluate_one_day_enhanced(
        hourly,
        band,
        df,
        REAL_DAY,
        grid_hw=GRID_HW,
        contamination=c,
        method="combined_persistence",
        calibrate_contamination_flag=False,
        truth_dilate_iters=1,
        combined_persistence_cfg=cp_cfg,
    )
    _, pred_cp = pairs[0]
    assert pred_cp.shape == fused.shape
