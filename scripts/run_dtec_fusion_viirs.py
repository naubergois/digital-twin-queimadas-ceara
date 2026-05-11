"""
Fusão GOES + VIIRS (DTEC §5) — mede o ganho de F1 em 2024-10-31.

Compara em R=5, 8, 10 km:
- DTEC GOES-only (baseline F1≈0.71)
- DTEC GOES + VIIRS fusion nos 4 modos (AND/OR/GATED/WEIGHTED)

VIIRS é (a) carregado de cache FIRMS se existir, (b) descarregado se
FIRMS_API_KEY estiver no ambiente, ou (c) sintetizado a partir dos focos
INPE como **proxy realista** (detection_rate=0.8, jitter 1.5 km, fp=0.01).

⚠️ Em modo proxy, a avaliação contra todos os focos INPE **sobrestima**
o ganho porque o VIIRS proxy é derivado da própria verdade. Reportamos
duas métricas:

1. ``F1_full``: contra todos os 76 focos (mede capacidade do pipeline)
2. ``F1_holdout``: contra os 20% de focos **não vistos** pelo VIIRS proxy
   (mede ganho real para focos novos)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure

from config.ceara_config import CEARA_BBOX
from src.dtec_supervised import (
    DTECSupervisedConfig,
    build_features,
    train_dtec_supervised,
)
from src.event_centric import day_window_utc, evaluate_event_centric
from src.firms_download import load_or_download_viirs
from src.multi_sensor_fusion import (
    FusionConfig,
    fuse_goes_with_viirs,
    synthesize_viirs_proxy,
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
    firms_cache = REPO_ROOT / "data" / "firms_cache"

    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())
    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)].reset_index(drop=True)

    # 1) Carregar GOES e treinar DTEC supervisionado
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
    peaks = model.predict_nms(feats, valid_bins, threshold=0.7258, nms_radius=1, smooth_sigma=1.2)
    pred_goes = _grow(peaks, valid_bins, 2)
    prob_goes = model.predict_proba_grid(feats, valid_bins)

    # 2) Obter VIIRS: cache → API → proxy sintético
    df_viirs = load_or_download_viirs(DAY, CEARA_BBOX, firms_cache)
    using_proxy = df_viirs.empty
    seed_for_holdout = None
    if using_proxy:
        seed_for_holdout = 13
        df_viirs = synthesize_viirs_proxy(
            df_day,
            bbox=CEARA_BBOX,
            detection_rate=0.80,
            spatial_jitter_km=1.5,
            false_positive_rate=0.01,
            n_cells_in_bbox=int(valid_bins.sum()),
            seed=seed_for_holdout,
        )
        # Reconstruir o conjunto holdout (focos NÃO vistos pelo VIIRS proxy)
        rng = np.random.default_rng(seed_for_holdout)
        keep = rng.random(len(df_day)) < 0.80
        df_holdout = df_day.iloc[np.flatnonzero(~keep)].reset_index(drop=True)
        print(f"⚠️ Modo proxy: VIIRS gerado de {len(df_day)} focos ({int(keep.sum())} detectados, jitter 1.5 km, 1% FP).")
        print(f"   Holdout (focos não vistos pelo VIIRS): {len(df_holdout)}/{len(df_day)}")
    else:
        df_holdout = pd.DataFrame(columns=df_day.columns)
        print(f"✅ VIIRS real carregado: {len(df_viirs)} detecções para {DAY}")

    # 3) Avaliação baseline (GOES only)
    print(f"\n=== Baseline GOES-only ===")
    for R in (5.0, 8.0, 10.0):
        m = evaluate_event_centric(
            pred_goes, df_day, CEARA_BBOX, GRID_HW,
            radius_km=R, valid_bins=valid_bins,
        )
        print(f"  R={R:>4.1f} km:  F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f}  ({int(pred_goes.sum())} cells)")

    # 4) Avaliação fundida nos 4 modos × 3 raios
    print(f"\n=== Fusão GOES + VIIRS (4 modos × 3 raios) ===")
    results = {"baseline": {}, "fused": []}
    for R in (5.0, 8.0, 10.0):
        m_b = evaluate_event_centric(
            pred_goes, df_day, CEARA_BBOX, GRID_HW,
            radius_km=R, valid_bins=valid_bins,
        )
        results["baseline"][f"R={R}km"] = {
            "f1": float(m_b.f1), "precision": float(m_b.precision), "recall": float(m_b.recall),
            "n_pred": int(pred_goes.sum()),
        }

    for mode in ("and", "or", "gated", "weighted"):
        for gate in (3.0, 5.0, 8.0):
            cfg_f = FusionConfig(mode=mode, gate_radius_km=gate)
            res = fuse_goes_with_viirs(
                pred_goes, prob_goes, df_viirs,
                CEARA_BBOX, GRID_HW, valid_bins,
                cfg=cfg_f, day_utc=(d0, d1),
            )
            for R in (5.0, 8.0, 10.0):
                m_full = evaluate_event_centric(
                    res.pred_mask, df_day, CEARA_BBOX, GRID_HW,
                    radius_km=R, valid_bins=valid_bins,
                )
                m_hold = None
                if using_proxy and len(df_holdout):
                    m_hold = evaluate_event_centric(
                        res.pred_mask, df_holdout, CEARA_BBOX, GRID_HW,
                        radius_km=R, valid_bins=valid_bins,
                    )
                row = {
                    "mode": mode, "gate_km": gate, "R_km": R,
                    "f1_full": float(m_full.f1),
                    "precision_full": float(m_full.precision),
                    "recall_full": float(m_full.recall),
                    "n_pred": int(res.pred_mask.sum()),
                    "f1_holdout": float(m_hold.f1) if m_hold is not None else None,
                    "precision_holdout": float(m_hold.precision) if m_hold is not None else None,
                    "recall_holdout": float(m_hold.recall) if m_hold is not None else None,
                }
                results["fused"].append(row)

    # Imprimir o melhor por modo em cada R
    print(f"\n{'mode':<10} {'gate':>5} {'R':>5} {'F1_full':>8} {'P_full':>7} {'R_full':>7} {'F1_hold':>8} {'#pred':>6}")
    by_mode_r = {}
    for r in results["fused"]:
        by_mode_r.setdefault((r["mode"], r["R_km"]), []).append(r)
    for (mode, R), rows in sorted(by_mode_r.items()):
        best = max(rows, key=lambda x: x["f1_full"])
        h = f"{best['f1_holdout']:.3f}" if best["f1_holdout"] is not None else "  —  "
        print(f"{mode:<10} {best['gate_km']:>5.1f} {R:>5.1f} {best['f1_full']:>8.3f} "
              f"{best['precision_full']:>7.3f} {best['recall_full']:>7.3f} {h:>8} {best['n_pred']:>6d}")

    # Vencedor global por F1_full e por F1_holdout
    print(f"\n=== TOP 3 globais por F1_full ===")
    for r in sorted(results["fused"], key=lambda x: -x["f1_full"])[:3]:
        print(json.dumps(r, indent=2))

    if using_proxy:
        print(f"\n=== TOP 3 globais por F1_holdout (mais honesto) ===")
        valid_rows = [r for r in results["fused"] if r["f1_holdout"] is not None]
        for r in sorted(valid_rows, key=lambda x: -x["f1_holdout"])[:3]:
            print(json.dumps(r, indent=2))

    out_path = REPO_ROOT / "data" / "goes16_eval" / "dtec_fusion_viirs_2024-10-31.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "using_proxy": using_proxy,
        "results": results,
    }, indent=2))
    print(f"\nGravado: {out_path}")


if __name__ == "__main__":
    main()
