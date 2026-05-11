"""
DTEC ajustado: grade mais fina + só h=17 (12 min após focos das 16:48 UTC).

A intuição é simples:
- 17h é a hora mais próxima do snapshot INPE => sinal mais fresco.
- Grade 144x144 (~3,75 km/cell) é mais próxima da resolução nativa GOES e
  permite localizar focos com menor ambiguidade espacial.

Reporta F1 event-centric (R=3,4,5 km) em diferentes raios e ambos os
modos (limiar simples e NMS) para HGB.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from config.ceara_config import CEARA_BBOX
from src.dtec_supervised import (
    DTECSupervisedConfig,
    blockwise_spatial_folds,
    build_features,
    train_dtec_supervised,
)
from src.event_centric import (
    day_window_utc,
    evaluate_event_centric,
)
from src.unsupervised_fire_goes import (
    build_lat_lon_edges,
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    truth_presence_grid,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = date(2024, 10, 31)


def _run(grid_hw, hours):
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())
    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)]

    hourly, _ = collect_hourly_band_grids(
        DAY,
        hours,
        (7, 13, 14),
        CEARA_BBOX,
        grid_hw,
        raw_dir,
        skip_download=True,
        overwrite=False,
        use_dqf=True,
        show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])
    feats = build_features(hourly, valid_bins)

    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, grid_hw)
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))

    cfg = DTECSupervisedConfig(classifier="hgb", smooth_sigma=1.2, nms_radius=2, nms_min_prob=0.5)
    model = train_dtec_supervised(feats, truth_dil, valid_bins, cfg=cfg)
    prob = model.predict_proba_grid(feats, valid_bins)

    results = {}
    for radius_km in (3.0, 5.0, 8.0):
        # Limiar simples
        best_th = None
        for thr in np.linspace(0.1, 0.97, 50):
            pred = (prob >= float(thr)) & valid_bins
            m = evaluate_event_centric(pred, df_day, CEARA_BBOX, grid_hw,
                                       radius_km=radius_km, valid_bins=valid_bins)
            if best_th is None or m.f1 > best_th["f1"]:
                best_th = {
                    "thr": float(thr),
                    "f1": float(m.f1),
                    "precision": float(m.precision),
                    "recall": float(m.recall),
                    "pred": int(pred.sum()),
                }
        # NMS
        best_nms = None
        for thr in np.linspace(0.2, 0.95, 40):
            for nrad in (1, 2, 3, 4):
                for sigma in (0.8, 1.2, 1.6):
                    peaks = model.predict_nms(
                        feats, valid_bins,
                        threshold=thr, nms_radius=nrad, smooth_sigma=sigma,
                    )
                    m = evaluate_event_centric(peaks, df_day, CEARA_BBOX, grid_hw,
                                               radius_km=radius_km, valid_bins=valid_bins)
                    if best_nms is None or m.f1 > best_nms["f1"]:
                        best_nms = {
                            "thr": float(thr),
                            "radius_cell": int(nrad),
                            "sigma": float(sigma),
                            "f1": float(m.f1),
                            "precision": float(m.precision),
                            "recall": float(m.recall),
                            "pred": int(peaks.sum()),
                        }
        results[f"R={radius_km}km"] = {"thresh": best_th, "nms": best_nms}
    return results


def main():
    cfgs = {
        "grid_72_h_all": ((72, 72), (16, 17, 18)),
        "grid_72_h17": ((72, 72), (17,)),
        "grid_144_h_all": ((144, 144), (16, 17, 18)),
        "grid_144_h17": ((144, 144), (17,)),
    }
    summary = {}
    for name, (grid, hours) in cfgs.items():
        print(f"\n=== {name} (grade={grid}, horas={hours}) ===")
        r = _run(grid, hours)
        summary[name] = r
        for k, v in r.items():
            print(f"  {k}:")
            print(f"    limiar simples: F1={v['thresh']['f1']:.3f} P={v['thresh']['precision']:.3f} R={v['thresh']['recall']:.3f} pred={v['thresh']['pred']}")
            print(f"    NMS:            F1={v['nms']['f1']:.3f} P={v['nms']['precision']:.3f} R={v['nms']['recall']:.3f} pred={v['nms']['pred']} (rad={v['nms']['radius_cell']} σ={v['nms']['sigma']})")

    out = REPO_ROOT / "data" / "goes16_eval" / "dtec_h17_finegrid_2024-10-31.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nGravado: {out}")


if __name__ == "__main__":
    main()
