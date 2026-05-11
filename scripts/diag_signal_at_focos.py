"""
GOES tem sinal nos focos INPE? Diagnóstico crítico antes de iterar o twin.

Sampleia BT13, BT7-BT14 e o score do gêmeo nos pixels onde o INPE colocou
focos, comparando com o resto da cena. Se os focos não tiverem assinatura
térmica acima da mediana, nenhum detector ABI vai recuperá-los — sinal real
para mudar a tática (ex.: snapshot horário oficial do INPE, AF L2 etc.).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from config.ceara_config import CEARA_BBOX
from src.event_centric import _build_grid_centers, _filter_focos, day_window_utc
from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig, hourly_anomaly_score
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


def _focos_grid_indices(df_day, bbox, grid_hw):
    h, w = grid_hw
    lat_edges, lon_edges = build_lat_lon_edges(bbox, grid_hw)
    rows = np.digitize(df_day["lat"].to_numpy(), lat_edges) - 1
    cols = np.digitize(df_day["lon"].to_numpy(), lon_edges) - 1
    keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return rows[keep], cols[keep]


def _percentile_rank(values: np.ndarray, ref_pool: np.ndarray) -> np.ndarray:
    """Para cada v em values, devolve a sua posição percentil em ref_pool [0..100]."""
    ref = np.sort(ref_pool)
    idx = np.searchsorted(ref, values, side="right")
    return 100.0 * idx / max(len(ref), 1)


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
    btd = bt7 - bt14

    twin = GOESFireDigitalTwin(bt13.shape, GOESFireDigitalTwinConfig(lof_neighbors=0))
    twin.ingest_series(hourly)
    risk = twin.smoothed_risk()

    rows, cols = _focos_grid_indices(df_day, CEARA_BBOX, GRID_HW)
    print(f"Focos com célula válida na grade {GRID_HW}: {len(rows)} de {len(df_day)}")

    # Pool de referência: só células válidas
    pool_bt13 = bt13[valid_bins]
    pool_btd = btd[valid_bins]
    pool_risk = risk[valid_bins]

    foco_bt13 = bt13[rows, cols]
    foco_btd = btd[rows, cols]
    foco_risk = risk[rows, cols]

    pr_bt13 = _percentile_rank(foco_bt13, pool_bt13)
    pr_btd = _percentile_rank(foco_btd, pool_btd)
    pr_risk = _percentile_rank(foco_risk, pool_risk)

    def _stats(name, arr):
        print(
            f"  {name}: min={arr.min():.2f} p25={np.percentile(arr,25):.2f} "
            f"med={np.median(arr):.2f} p75={np.percentile(arr,75):.2f} max={arr.max():.2f}"
        )

    print("\nValor absoluto BT13(K) max-cena nas células dos focos:")
    _stats("focos ", foco_bt13)
    _stats("cena  ", pool_bt13)
    print(f"  diferença mediana = {np.median(foco_bt13) - np.median(pool_bt13):+.2f} K")

    print("\nValor absoluto BT7-BT14 (K) nas células dos focos:")
    _stats("focos ", foco_btd)
    _stats("cena  ", pool_btd)
    print(f"  diferença mediana = {np.median(foco_btd) - np.median(pool_btd):+.2f} K")

    print("\nRisco do gêmeo (0–1, após smoothed) nas células dos focos:")
    _stats("focos ", foco_risk)
    _stats("cena  ", pool_risk)

    print("\nPercentil de cada foco no pool da cena (100 = topo):")
    _stats("rank BT13", pr_bt13)
    _stats("rank BTD ", pr_btd)
    _stats("rank risk", pr_risk)

    n_top1 = int(np.sum(pr_risk >= 99))
    n_top2 = int(np.sum(pr_risk >= 98))
    n_top5 = int(np.sum(pr_risk >= 95))
    print(f"\nFocos com risco no top 1%: {n_top1}/{len(rows)}")
    print(f"Focos com risco no top 2%: {n_top2}/{len(rows)}")
    print(f"Focos com risco no top 5%: {n_top5}/{len(rows)}")


if __name__ == "__main__":
    main()
