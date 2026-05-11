"""
Treina e avalia o DTEC supervisionado com validação por blocos espaciais (DTEC §6).

Reporta:
- F1 event-centric *in-sample* (toda a cena) — para diagnóstico.
- F1 event-centric *held-out* por bloco (3×3 = 9 folds) — métrica honesta.
- Importâncias dos coeficientes (lê o classificador).
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
GRID_HW = (72, 72)


def main() -> None:
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())

    hourly, _ = collect_hourly_band_grids(
        DAY,
        (16, 17, 18),
        (7, 13, 14),
        CEARA_BBOX,
        GRID_HW,
        raw_dir,
        skip_download=True,
        overwrite=False,
        use_dqf=True,
        show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])
    feats = build_features(hourly, valid_bins)

    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)]
    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, GRID_HW)
    # Para o treino: dilatamos a verdade 1 célula (tolerância à posição do foco
    # dentro do bin de 7 km). Não é usado na avaliação — essa é event-centric.
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))

    # === Avaliação in-sample (sanity) ===
    for clf_kind in ("logreg", "hgb"):
        cfg = DTECSupervisedConfig(
            C=0.5,
            threshold=0.5,
            classifier=clf_kind,
            smooth_sigma=1.0,
            nms_radius=2,
            nms_min_prob=0.5,
        )
        model = train_dtec_supervised(feats, truth_dil, valid_bins, cfg=cfg)
        prob = model.predict_proba_grid(feats, valid_bins)

        # Mode 1: limiar simples
        best_th = None
        for thr in np.linspace(0.1, 0.97, 60):
            pred = (prob >= float(thr)) & valid_bins
            m = evaluate_event_centric(
                pred,
                df_day,
                CEARA_BBOX,
                GRID_HW,
                radius_km=5.0,
                valid_bins=valid_bins,
            )
            if best_th is None or m.f1 > best_th["f1"]:
                best_th = {
                    "thr": float(thr),
                    "f1": float(m.f1),
                    "precision": float(m.precision),
                    "recall": float(m.recall),
                    "pred_cells": int(pred.sum()),
                }

        # Mode 2: NMS
        best_nms = None
        for thr in np.linspace(0.2, 0.95, 40):
            for radius in (1, 2, 3):
                peaks = model.predict_nms(
                    feats,
                    valid_bins,
                    threshold=thr,
                    nms_radius=radius,
                    smooth_sigma=1.0,
                )
                m = evaluate_event_centric(
                    peaks,
                    df_day,
                    CEARA_BBOX,
                    GRID_HW,
                    radius_km=5.0,
                    valid_bins=valid_bins,
                )
                if best_nms is None or m.f1 > best_nms["f1"]:
                    best_nms = {
                        "thr": float(thr),
                        "radius": int(radius),
                        "f1": float(m.f1),
                        "precision": float(m.precision),
                        "recall": float(m.recall),
                        "pred_cells": int(peaks.sum()),
                    }

        print(f"\n=== In-sample [{clf_kind}] (R=5 km) ===")
        print(f"limiar simples: {json.dumps(best_th)}")
        print(f"NMS (picos):    {json.dumps(best_nms)}")

    # === Bloco-spatial CV com NMS ===
    cfg_best = DTECSupervisedConfig(
        classifier="hgb",
        smooth_sigma=1.0,
        nms_radius=2,
        nms_min_prob=0.5,
    )
    folds = blockwise_spatial_folds(GRID_HW, n_blocks_lat=3, n_blocks_lon=3)
    fold_metrics = []
    print("\n=== Bloco-spatial 3x3 (HGB + NMS, R=5 km) ===")
    print(f"{'fold':>4} {'#foc_test':>10} {'#pred':>6} {'P':>6} {'R':>6} {'F1':>6} {'thr':>5} {'rad':>4}")
    for k, (train_mask, test_mask) in enumerate(folds):
        mdl = train_dtec_supervised(
            feats, truth_dil, valid_bins, cfg=cfg_best, train_mask=train_mask
        )
        lat_edges, lon_edges = build_lat_lon_edges(CEARA_BBOX, GRID_HW)
        rows = np.digitize(df_day["lat"].to_numpy(), lat_edges) - 1
        cols = np.digitize(df_day["lon"].to_numpy(), lon_edges) - 1
        keep = (
            (rows >= 0)
            & (rows < GRID_HW[0])
            & (cols >= 0)
            & (cols < GRID_HW[1])
            & test_mask[np.clip(rows, 0, GRID_HW[0] - 1), np.clip(cols, 0, GRID_HW[1] - 1)]
        )
        df_test = df_day.iloc[keep.nonzero()[0]] if keep.any() else df_day.iloc[:0]
        valid_test = valid_bins & test_mask

        best_fold = None
        for thr in np.linspace(0.2, 0.95, 32):
            for radius in (1, 2, 3):
                peaks = mdl.predict_nms(
                    feats,
                    valid_bins,
                    threshold=float(thr),
                    nms_radius=radius,
                    smooth_sigma=1.0,
                )
                peaks_test = peaks & valid_test
                m = evaluate_event_centric(
                    peaks_test,
                    df_test,
                    CEARA_BBOX,
                    GRID_HW,
                    radius_km=5.0,
                    valid_bins=valid_test,
                )
                if best_fold is None or m.f1 > best_fold["f1"]:
                    best_fold = {
                        "thr": float(thr),
                        "radius": int(radius),
                        "f1": float(m.f1),
                        "precision": float(m.precision),
                        "recall": float(m.recall),
                        "n_pred": int(peaks_test.sum()),
                        "n_focos": int(m.n_focos),
                    }
        fold_metrics.append(best_fold)
        print(
            f"{k:>4d} {best_fold['n_focos']:>10d} {best_fold['n_pred']:>6d} "
            f"{best_fold['precision']:>6.3f} {best_fold['recall']:>6.3f} "
            f"{best_fold['f1']:>6.3f} {best_fold['thr']:>5.2f} {best_fold['radius']:>4d}"
        )

    nonzero = [m for m in fold_metrics if m["n_focos"] > 0]
    if nonzero:
        f1s = np.array([m["f1"] for m in nonzero])
        print(f"\nF1 médio (folds com focos, n={len(nonzero)}): {f1s.mean():.3f} ± {f1s.std():.3f}")
        print(f"F1 mediano: {float(np.median(f1s)):.3f}")
    out_path = REPO_ROOT / "data" / "goes16_eval" / "dtec_supervised_2024-10-31.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"folds": fold_metrics}, indent=2))
    print(f"\nGravado: {out_path}")


if __name__ == "__main__":
    main()
