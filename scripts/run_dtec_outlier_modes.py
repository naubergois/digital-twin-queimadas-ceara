"""
Outliers + gêmeo digital: vários modos de combinação.

- ``AND``       — baseline ∩ outliers (alto P, R baixo) — testado em run_dtec_outlier_layer.
- ``ONLY``      — outliers como detector primário (sem supervisão).
- ``UNION``     — baseline ∪ outliers (recupera focos perdidos por NMS).
- ``UNION_NMS`` — baseline ∪ outliers, depois NMS+dilatação espacial.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, maximum_filter, gaussian_filter

from config.ceara_config import CEARA_BBOX
from src.dtec_outlier import OutlierConfig, outlier_mask_from_twin_features
from src.dtec_supervised import (
    DTECSupervisedConfig,
    build_features,
    train_dtec_supervised,
)
from src.event_centric import (
    day_window_utc,
    evaluate_event_centric,
)
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    truth_presence_grid,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = date(2024, 10, 31)
GRID_HW = (144, 144)
HOURS = (16, 17, 18)


def _grow(peaks, valid_bins, n_iters):
    if n_iters <= 0:
        return peaks
    struct = generate_binary_structure(2, 2)
    d = peaks.copy()
    for _ in range(n_iters):
        d = binary_dilation(d, structure=struct)
    return d & valid_bins


def main():
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())
    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)]

    hourly, _ = collect_hourly_band_grids(
        DAY, HOURS, (7, 13, 14), CEARA_BBOX, GRID_HW, raw_dir,
        skip_download=True, overwrite=False, use_dqf=True, show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])
    feats = build_features(hourly, valid_bins)

    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, GRID_HW)
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))

    cfg = DTECSupervisedConfig(classifier="hgb", smooth_sigma=1.2, nms_radius=1, nms_min_prob=0.5)
    model = train_dtec_supervised(feats, truth_dil, valid_bins, cfg=cfg)

    peaks = model.predict_nms(
        feats, valid_bins, threshold=0.7258, nms_radius=1, smooth_sigma=1.2,
    )
    pred_base = _grow(peaks, valid_bins, 2)

    R = 10.0
    m_base = evaluate_event_centric(
        pred_base, df_day, CEARA_BBOX, GRID_HW, radius_km=R, valid_bins=valid_bins,
    )
    print(f"\nBaseline:  F1={m_base.f1:.3f} P={m_base.precision:.3f} R={m_base.recall:.3f} pred={int(pred_base.sum())}")

    summary = []

    print("\n=== ONLY (outliers como detector primário) ===")
    for method in ("isolation_forest", "local_outlier_factor", "elliptic_envelope", "ensemble"):
        for cont in (0.005, 0.01, 0.02, 0.04, 0.08):
            ocfg = OutlierConfig(method=method, contamination=cont)
            out = outlier_mask_from_twin_features(feats, valid_bins, cfg=ocfg)
            # NMS sobre outliers para reduzir clusters densos
            for dil in (0, 1, 2):
                pred = _grow(out, valid_bins, dil)
                m = evaluate_event_centric(
                    pred, df_day, CEARA_BBOX, GRID_HW, radius_km=R, valid_bins=valid_bins,
                )
                summary.append({"mode": "ONLY", "method": method, "cont": cont, "dilate": dil,
                                "f1": float(m.f1), "precision": float(m.precision), "recall": float(m.recall),
                                "n_pred": int(pred.sum())})
        rows = [s for s in summary if s["mode"]=="ONLY" and s["method"]==method]
        best = max(rows, key=lambda x: x["f1"])
        print(f"  {method:<24} F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} cont={best['cont']} dil={best['dilate']} pred={best['n_pred']}")

    print("\n=== UNION (baseline ∪ outliers) ===")
    for method in ("isolation_forest", "local_outlier_factor", "ensemble"):
        for cont in (0.005, 0.01, 0.02, 0.04, 0.08):
            ocfg = OutlierConfig(method=method, contamination=cont)
            out = outlier_mask_from_twin_features(feats, valid_bins, cfg=ocfg)
            for dil_out in (0, 1, 2):
                out_grown = _grow(out, valid_bins, dil_out)
                pred = pred_base | out_grown
                m = evaluate_event_centric(
                    pred, df_day, CEARA_BBOX, GRID_HW, radius_km=R, valid_bins=valid_bins,
                )
                summary.append({"mode": "UNION", "method": method, "cont": cont, "dilate_out": dil_out,
                                "f1": float(m.f1), "precision": float(m.precision), "recall": float(m.recall),
                                "n_pred": int(pred.sum())})
        rows = [s for s in summary if s["mode"]=="UNION" and s["method"]==method]
        best = max(rows, key=lambda x: x["f1"])
        print(f"  {method:<24} F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} cont={best['cont']} dil_out={best.get('dilate_out')} pred={best['n_pred']}")

    print("\n=== INTERSEÇÃO PONDERADA (outlier_mask × prob_smoothed) ===")
    # Multiplica máscara de outliers pelo prob suavizado; pega top-k células
    prob = model.predict_proba_grid(feats, valid_bins)
    prob_s = gaussian_filter(prob, sigma=1.2, mode="nearest")
    for method in ("isolation_forest", "local_outlier_factor", "ensemble"):
        for cont in (0.01, 0.02, 0.04, 0.08):
            ocfg = OutlierConfig(method=method, contamination=cont)
            out = outlier_mask_from_twin_features(feats, valid_bins, cfg=ocfg)
            score = np.where(out, prob_s, 0.0)
            # NMS sobre score
            for nrad in (1, 2, 3):
                local_max = maximum_filter(score, size=2*nrad+1, mode="nearest")
                peaks_s = (score == local_max) & (score > 0) & out & valid_bins
                for dil in (0, 1, 2):
                    pred = _grow(peaks_s, valid_bins, dil)
                    m = evaluate_event_centric(
                        pred, df_day, CEARA_BBOX, GRID_HW, radius_km=R, valid_bins=valid_bins,
                    )
                    summary.append({"mode": "WEIGHT_NMS", "method": method, "cont": cont, "nrad": nrad, "dilate": dil,
                                    "f1": float(m.f1), "precision": float(m.precision), "recall": float(m.recall),
                                    "n_pred": int(pred.sum())})
        rows = [s for s in summary if s["mode"]=="WEIGHT_NMS" and s["method"]==method]
        best = max(rows, key=lambda x: x["f1"])
        print(f"  {method:<24} F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} cont={best['cont']} nrad={best.get('nrad')} dil={best.get('dilate')} pred={best['n_pred']}")

    print(f"\n=== TOP 5 globais (R={R} km) ===")
    summary.sort(key=lambda x: -x["f1"])
    for s in summary[:5]:
        print(json.dumps(s, indent=2))

    out_path = REPO_ROOT / "data" / "goes16_eval" / "dtec_outlier_modes_2024-10-31.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline": {
            "f1": float(m_base.f1),
            "precision": float(m_base.precision),
            "recall": float(m_base.recall),
            "n_pred": int(pred_base.sum()),
        },
        "all": summary,
    }, indent=2))
    print(f"\nGravado: {out_path}")


if __name__ == "__main__":
    main()
