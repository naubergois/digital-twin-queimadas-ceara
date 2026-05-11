"""
Camada de outliers sobre o estado do gêmeo digital: melhora precisão?

Compara, em 2024-10-31 e R=10 km, o pipeline DTEC (HGB+NMS+dilatação)
**sem** vs **com** filtro de outlier sobre as features do twin.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from config.ceara_config import CEARA_BBOX
from src.dtec_outlier import OutlierConfig, filter_predictions_by_outlier
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

    # Configuração-âncora do final_push: thr=0.726, nrad=1, sigma=1.2, dilate=2
    peaks = model.predict_nms(
        feats, valid_bins,
        threshold=0.7258, nms_radius=1, smooth_sigma=1.2,
    )
    pred_base = _grow(peaks, valid_bins, 2)

    # Avaliação baseline
    m_base = evaluate_event_centric(
        pred_base, df_day, CEARA_BBOX, GRID_HW,
        radius_km=10.0, valid_bins=valid_bins,
    )

    print(f"\n=== Baseline DTEC (HGB+NMS+dil 2) — R=10 km ===")
    print(f"F1={m_base.f1:.3f} P={m_base.precision:.3f} R={m_base.recall:.3f} pred={int(pred_base.sum())}")

    print(f"\n=== Com filtro de outliers sobre features do gêmeo digital ===")
    out_results = []
    for method in ("isolation_forest", "local_outlier_factor", "elliptic_envelope", "ensemble"):
        for cont in (0.005, 0.01, 0.02, 0.04, 0.08, 0.12):
            ocfg = OutlierConfig(method=method, contamination=cont)
            pred = filter_predictions_by_outlier(pred_base, feats, valid_bins, cfg=ocfg)
            for R in (5.0, 8.0, 10.0):
                m = evaluate_event_centric(
                    pred, df_day, CEARA_BBOX, GRID_HW,
                    radius_km=R, valid_bins=valid_bins,
                )
                out_results.append({
                    "method": method,
                    "contamination": cont,
                    "R_km": R,
                    "f1": float(m.f1),
                    "precision": float(m.precision),
                    "recall": float(m.recall),
                    "n_pred": int(pred.sum()),
                })

    # Mostrar o melhor por método ao R=10 km
    print(f"\n{'method':<24} {'cont':>6} {'F1':>6} {'P':>6} {'R':>6} {'#pred':>6}")
    by_method = {}
    for r in out_results:
        if r["R_km"] != 10.0:
            continue
        by_method.setdefault(r["method"], []).append(r)
    for method, rows in by_method.items():
        best = max(rows, key=lambda x: x["f1"])
        print(f"{method:<24} {best['contamination']:>6.3f} {best['f1']:>6.3f} {best['precision']:>6.3f} {best['recall']:>6.3f} {best['n_pred']:>6d}")

    # Curva completa por R para o melhor método global
    best_global = max(out_results, key=lambda x: x["f1"])
    print(f"\n=== Melhor combinação (qualquer R) ===")
    print(json.dumps(best_global, indent=2))

    out_path = REPO_ROOT / "data" / "goes16_eval" / "dtec_outlier_layer_2024-10-31.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline": {
            "f1": float(m_base.f1),
            "precision": float(m_base.precision),
            "recall": float(m_base.recall),
            "n_pred": int(pred_base.sum()),
        },
        "with_outlier_filter": out_results,
    }, indent=2))
    print(f"\nGravado: {out_path}")


if __name__ == "__main__":
    main()
