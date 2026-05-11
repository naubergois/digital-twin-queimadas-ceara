"""
Constrói mapas interativos DTEC (HTML, Folium) para uma ou mais datas.

Para cada data:
1. Carrega NetCDFs locais + focos INPE.
2. Treina HGB sobre features do gêmeo digital.
3. Aplica NMS + dilatação morfológica → máscara de previsão.
4. Grava ``mapa_AAAA-MM-DD.html`` com:
   - Bbox do Ceará
   - Focos INPE (verde = TP, vermelho = FN)
   - Previsões DTEC (verde = TP, laranja = FP)
   - Legenda com métricas event-centric

Uso: ``python -m scripts.build_dtec_maps``

Por defeito gera 2024-10-31; ajuste ``DATES`` para listas multi-dia
quando houver mais granulos locais.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from config.ceara_config import CEARA_BBOX
from src.dtec_outlier import OutlierConfig, filter_predictions_by_outlier
from src.dtec_supervised import (
    DTECSupervisedConfig,
    build_features,
    train_dtec_supervised,
)
from src.event_centric import day_window_utc
from src.map_view import build_multi_date_index, save_map
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    truth_presence_grid,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATES = (date(2024, 10, 31),)
GRID_HW = (144, 144)
HOURS = (16, 17, 18)
R_KM = 10.0


def _grow(peaks, valid_bins, n_iters):
    if n_iters <= 0:
        return peaks
    struct = generate_binary_structure(2, 2)
    d = peaks.copy()
    for _ in range(n_iters):
        d = binary_dilation(d, structure=struct)
    return d & valid_bins


def _predict_for_day(day_utc: date, df, raw_dir: Path):
    hourly, _ = collect_hourly_band_grids(
        day_utc, HOURS, (7, 13, 14), CEARA_BBOX, GRID_HW, raw_dir,
        skip_download=True, overwrite=False, use_dqf=True, show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])
    feats = build_features(hourly, valid_bins)

    d0, d1 = day_window_utc(day_utc.isoformat())
    df_day = df.loc[(df["datetime"] >= d0) & (df["datetime"] < d1)]

    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, GRID_HW)
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))

    cfg = DTECSupervisedConfig(classifier="hgb", smooth_sigma=1.2, nms_radius=1, nms_min_prob=0.5)
    model = train_dtec_supervised(feats, truth_dil, valid_bins, cfg=cfg)

    peaks = model.predict_nms(
        feats, valid_bins, threshold=0.7258, nms_radius=1, smooth_sigma=1.2,
    )
    pred_f1 = _grow(peaks, valid_bins, 2)

    # Modo precisão-focado: AND com outliers (LOF cont=0.12)
    pred_precision = filter_predictions_by_outlier(
        pred_f1, feats, valid_bins,
        cfg=OutlierConfig(method="local_outlier_factor", contamination=0.12),
    )

    return df_day, pred_f1, pred_precision, valid_bins


def main(dates: Sequence[date] = DATES):
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    map_dir = REPO_ROOT / "data" / "goes16_eval" / "maps_html"
    map_dir.mkdir(parents=True, exist_ok=True)

    df = load_inpe_focos(inpe_csv)
    pairs = []
    for d in dates:
        print(f"\n=== {d} ===")
        df_day, pred_f1, pred_p, valid_bins = _predict_for_day(d, df, raw_dir)
        print(f"focos no dia: {len(df_day)} | pred F1-mode: {int(pred_f1.sum())} | pred precision-mode: {int(pred_p.sum())}")

        out_f1 = map_dir / f"mapa_{d.isoformat()}_F1.html"
        save_map(out_f1, df_day, pred_f1, CEARA_BBOX, GRID_HW,
                 day_iso=d.isoformat(), radius_km=R_KM, valid_bins=valid_bins,
                 title="DTEC (modo F1)")
        print(f"  F1 mode  → {out_f1}")

        out_p = map_dir / f"mapa_{d.isoformat()}_precisao.html"
        save_map(out_p, df_day, pred_p, CEARA_BBOX, GRID_HW,
                 day_iso=d.isoformat(), radius_km=R_KM, valid_bins=valid_bins,
                 title="DTEC (modo precisão, outlier filter LOF cont=0.12)")
        print(f"  precision→ {out_p}")

        pairs.append((f"{d.isoformat()} (F1)", out_f1))
        pairs.append((f"{d.isoformat()} (precisão)", out_p))

    index = build_multi_date_index(pairs, map_dir / "index.html")
    print(f"\nÍndice: {index}")


if __name__ == "__main__":
    main()
