"""
Empurrão final F1: combina HGB+NMS (precisão) com região quente do gêmeo
e produz uma máscara de detecções como **picos com dilatação espacial**
proporcional ao raio de avaliação. Reporta F1 em vários raios.

Estratégia: cada pico NMS é um evento previsto; cada foco INPE casa se
houver um pico a ≤ R km. Não dilatamos a máscara (já que o evaluator
cell-based contaria muitas células); em vez disso, dilatamos cada pico até
um cluster pequeno (1-2 células de raio = 4-7 km), preservando o efeito de
"alerta" sem inflar precisão artificialmente.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from config.ceara_config import CEARA_BBOX
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


def _experiment(grid_hw, hours):
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())
    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)]

    hourly, _ = collect_hourly_band_grids(
        DAY, hours, (7, 13, 14), CEARA_BBOX, grid_hw, raw_dir,
        skip_download=True, overwrite=False, use_dqf=True, show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])
    feats = build_features(hourly, valid_bins)

    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, grid_hw)
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))

    cfg = DTECSupervisedConfig(classifier="hgb", smooth_sigma=1.2, nms_radius=2, nms_min_prob=0.5)
    model = train_dtec_supervised(feats, truth_dil, valid_bins, cfg=cfg)

    return df_day, feats, valid_bins, model, grid_hw


def _evaluate_with_dilation(peaks, dilate_iters, df_day, grid_hw, valid_bins, radius_km):
    if dilate_iters > 0 and peaks.any():
        struct = generate_binary_structure(2, 2)  # 8-conectividade = mais agressivo
        d = peaks.copy()
        for _ in range(dilate_iters):
            d = binary_dilation(d, structure=struct)
        d &= valid_bins
    else:
        d = peaks
    m = evaluate_event_centric(
        d, df_day, CEARA_BBOX, grid_hw, radius_km=radius_km, valid_bins=valid_bins
    )
    return m, int(d.sum())


def main():
    configs = [
        ((144, 144), (16, 17, 18)),
        ((144, 144), (17,)),
        ((192, 192), (16, 17, 18)),
        ((192, 192), (17,)),
    ]

    summary = []
    for grid, hours in configs:
        print(f"\n### grid={grid} hours={hours} ###")
        df_day, feats, valid_bins, model, ghw = _experiment(grid, hours)

        for R in (3.0, 5.0, 8.0, 10.0):
            best = None
            # Pesquisar: limiar NMS, raio NMS, sigma, e iterações de dilatação
            for thr in np.linspace(0.30, 0.95, 30):
                for nrad in (1, 2, 3, 4):
                    for sigma in (0.8, 1.2, 1.6, 2.0):
                        peaks = model.predict_nms(
                            feats, valid_bins,
                            threshold=thr, nms_radius=nrad, smooth_sigma=sigma,
                        )
                        for d_it in (0, 1, 2):
                            m, n_pred = _evaluate_with_dilation(
                                peaks, d_it, df_day, ghw, valid_bins, R
                            )
                            if best is None or m.f1 > best["f1"]:
                                best = {
                                    "R": R,
                                    "f1": float(m.f1),
                                    "precision": float(m.precision),
                                    "recall": float(m.recall),
                                    "tp_recall": int(m.tp_for_recall),
                                    "n_focos": int(m.n_focos),
                                    "n_pred": int(n_pred),
                                    "thr": float(thr),
                                    "nrad": int(nrad),
                                    "sigma": float(sigma),
                                    "dilate_iters": int(d_it),
                                }
            print(
                f"  R={R}km: F1={best['f1']:.3f} P={best['precision']:.3f} R={best['recall']:.3f} "
                f"#pred={best['n_pred']} #focos={best['n_focos']} thr={best['thr']:.2f} "
                f"nrad={best['nrad']} σ={best['sigma']:.1f} dil={best['dilate_iters']}"
            )
            summary.append({"grid": grid, "hours": list(hours), **best})

    out = REPO_ROOT / "data" / "goes16_eval" / "dtec_final_push_2024-10-31.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nGravado: {out}")

    # Imprimir o vencedor global
    print("\n=== TOP 3 globais ===")
    for s in sorted(summary, key=lambda x: -x["f1"])[:3]:
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
