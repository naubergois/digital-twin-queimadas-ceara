"""
Mede F1 event-centric (DTEC §4) de cada detector existente em 2024-10-31.

Não altera o pipeline; só anexa métricas event-centric ao lado das de grade,
para diagnosticar o ponto de partida e guiar a iteração do gêmeo digital.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from config.ceara_config import CEARA_BBOX
from src.event_centric import (
    day_window_utc,
    evaluate_event_centric,
    evaluate_event_centric_multi_radius,
)
from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig
from src.goes_fire_method_v2 import CombinedPersistenceConfig, detect_combined_persistence
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    confusion,
    detect_isolation_forest_multiband,
    detect_spatial_residual_multiband,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    merge_band_grids_max,
    truth_presence_grid,
    dilate_truth_grid,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = date(2024, 10, 31)
HOURS = (16, 17, 18)
CHANNELS = (7, 13, 14)
GRID_HW = (72, 72)


def main() -> None:
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"

    df = load_inpe_focos(inpe_csv)
    hourly, _ = collect_hourly_band_grids(
        DAY,
        HOURS,
        CHANNELS,
        CEARA_BBOX,
        GRID_HW,
        raw_dir,
        skip_download=True,
        overwrite=False,
        use_dqf=True,
        show_progress=False,
    )
    band = merge_band_grids_max(hourly, CHANNELS)
    bt13 = band[13]
    bt7 = band.get(7)
    bt14 = band.get(14)
    valid_bins = intersect_valid_bins_hourly(hourly, sorted(CHANNELS))

    d0, d1 = day_window_utc(DAY.isoformat())
    truth_raw = truth_presence_grid(
        df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)],
        CEARA_BBOX,
        GRID_HW,
    )
    truth_d1 = dilate_truth_grid(truth_raw, 1)

    # Lista de configurações a comparar.
    twin_cfg = GOESFireDigitalTwinConfig(
        persistence=0.5,
        gaussian_sigma=1.2,
        dbt_weight=0.55,
        multiscale_median_sizes=(5, 9, 15),
        fusion="prob_or",
        threshold_mode="percentile",
        lof_neighbors=24,
    )

    cp_cfg = CombinedPersistenceConfig()

    methods = {}

    for c in (0.01, 0.02, 0.04, 0.06):
        methods[f"spatial_residual_c{c}"] = detect_spatial_residual_multiband(
            bt13, bt7, bt14, contamination=c
        )
        methods[f"isolation_forest_c{c}"] = detect_isolation_forest_multiband(
            bt7, bt13, bt14, contamination=c
        )
        methods[f"combined_persistence_c{c}"] = detect_combined_persistence(
            hourly, valid_bins, c, cfg=cp_cfg
        )
        twin = GOESFireDigitalTwin(bt13.shape, twin_cfg)
        twin.ingest_series(hourly)
        methods[f"digital_twin_c{c}"] = twin.predict_mask(c, valid_bins)

    results = {}
    for name, pred in methods.items():
        # Métricas de grade (current)
        grid_m = confusion(pred, truth_d1, valid_bins)
        # Event-centric: curva por raio
        ec_curve = evaluate_event_centric_multi_radius(
            pred,
            df,
            CEARA_BBOX,
            GRID_HW,
            radii_km=(1.5, 3.0, 5.0, 8.0),
            day_utc=(d0, d1),
            valid_bins=valid_bins,
        )
        results[name] = {
            "grid": {k: round(float(v), 4) for k, v in grid_m.items()},
            "event_centric": {
                rk: {k: round(float(v), 4) for k, v in rv.items()} for rk, rv in ec_curve.items()
            },
            "pred_cells": int(pred.sum()),
        }

    out_path = REPO_ROOT / "data" / "goes16_eval" / "dtec_baseline_2024-10-31.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nResultado completo: {out_path}\n")
    print(f"{'método':<32} {'grid F1':>8} {'EC F1 (R=3km)':>14} {'EC F1 (R=5km)':>14} {'EC F1 (R=8km)':>14} {'#focos':>8} {'#comps':>8}")
    for name, r in results.items():
        gf1 = r["grid"]["f1"]
        e3 = r["event_centric"]["R=3.0km"]
        e5 = r["event_centric"]["R=5.0km"]
        e8 = r["event_centric"]["R=8.0km"]
        print(
            f"{name:<32} {gf1:>8.3f} {e3['ec_f1']:>14.3f} {e5['ec_f1']:>14.3f} "
            f"{e8['ec_f1']:>14.3f} {int(e3['ec_n_focos']):>8d} {int(e3['ec_n_components']):>8d}"
        )


if __name__ == "__main__":
    main()
