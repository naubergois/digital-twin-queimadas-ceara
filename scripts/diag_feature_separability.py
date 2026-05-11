"""Há sinal multi-feature discriminativo nos focos? Se sim, regressão logística resolve."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, binary_dilation

from config.ceara_config import CEARA_BBOX
from src.event_centric import day_window_utc, _filter_focos
from src.unsupervised_fire_goes import (
    build_lat_lon_edges,
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
    merge_band_grids_max,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = date(2024, 10, 31)
GRID_HW = (72, 72)


def main() -> None:
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())
    df_day = _filter_focos(df, CEARA_BBOX, day_utc=(d0, d1))

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
    band = merge_band_grids_max(hourly, (7, 13, 14))
    bt13 = band[13]
    bt7 = band[7]
    bt14 = band[14]

    # Features por célula
    def _fill(x):
        return np.where(np.isfinite(x), x, np.nanmedian(x))

    f13 = _fill(bt13)
    f7 = _fill(bt7)
    f14 = _fill(bt14)

    bg13_5 = median_filter(f13, size=5, mode="nearest")
    bg13_11 = median_filter(f13, size=11, mode="nearest")
    bg13_21 = median_filter(f13, size=21, mode="nearest")
    res13_5 = f13 - bg13_5
    res13_11 = f13 - bg13_11
    res13_21 = f13 - bg13_21
    btd = f7 - f14
    bg_btd_11 = median_filter(btd, size=11, mode="nearest")
    res_btd = btd - bg_btd_11

    # Persistência: anomalia positiva por hora em escala 5
    n_active_h = np.zeros_like(f13)
    for slot in hourly:
        b = _fill(slot[13])
        bg = median_filter(b, size=5, mode="nearest")
        n_active_h += (b - bg > 0.5).astype(np.float64)

    truth_cells = np.zeros(GRID_HW, dtype=bool)
    lat_edges, lon_edges = build_lat_lon_edges(CEARA_BBOX, GRID_HW)
    rows = np.digitize(df_day["lat"].to_numpy(), lat_edges) - 1
    cols = np.digitize(df_day["lon"].to_numpy(), lon_edges) - 1
    keep = (rows >= 0) & (rows < GRID_HW[0]) & (cols >= 0) & (cols < GRID_HW[1])
    rows, cols = rows[keep], cols[keep]
    truth_cells[rows, cols] = True
    # Dilatação 1 célula → tolerância
    truth_dilated = binary_dilation(truth_cells, structure=np.ones((3, 3), dtype=bool))

    pos = truth_dilated & valid_bins
    neg = (~truth_dilated) & valid_bins

    print(f"Cells positivas (após dilatação 1): {int(pos.sum())} | negativas: {int(neg.sum())}")

    feats = {
        "bt13": f13,
        "res13_5": res13_5,
        "res13_11": res13_11,
        "res13_21": res13_21,
        "btd": btd,
        "res_btd_11": res_btd,
        "bt7": f7,
        "n_active_h": n_active_h,
    }

    print(f"\n{'feature':<15} {'med_pos':>10} {'med_neg':>10} {'p75_pos':>10} {'p75_neg':>10} {'p_separ':>10}")
    for name, F in feats.items():
        pp = F[pos]
        nn = F[neg]
        med_p = float(np.median(pp))
        med_n = float(np.median(nn))
        p75_p = float(np.percentile(pp, 75))
        p75_n = float(np.percentile(nn, 75))
        # Estatística de separação simples: AUC pseudo via mediana padronizada
        std_n = float(np.std(nn) + 1e-9)
        sep = (med_p - med_n) / std_n
        print(f"{name:<15} {med_p:>10.3f} {med_n:>10.3f} {p75_p:>10.3f} {p75_n:>10.3f} {sep:>10.3f}")


if __name__ == "__main__":
    main()
