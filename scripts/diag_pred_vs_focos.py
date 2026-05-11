"""Inspecção rápida: onde estão focos vs centróides previstos em 2024-10-31."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from config.ceara_config import CEARA_BBOX
from src.event_centric import (
    _build_grid_centers,
    _components_centroids,
    _filter_focos,
    _haversine_like_km,
    day_window_utc,
)
from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
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
    print(f"Focos INPE no Ceará em {DAY}: {len(df_day)}")
    print(df_day[["datetime", "lat", "lon"]].head(8).to_string(index=False))
    print()
    print(f"BBOX lat=[{CEARA_BBOX['min_lat']},{CEARA_BBOX['max_lat']}] lon=[{CEARA_BBOX['min_lon']},{CEARA_BBOX['max_lon']}]")
    print(f"lat focos: [{df_day['lat'].min():.3f}, {df_day['lat'].max():.3f}]")
    print(f"lon focos: [{df_day['lon'].min():.3f}, {df_day['lon'].max():.3f}]")

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
    twin = GOESFireDigitalTwin(hourly[0][13].shape, GOESFireDigitalTwinConfig(lof_neighbors=0))
    twin.ingest_series(hourly)
    pred = twin.predict_mask(0.04, valid_bins)
    print(f"\ntwin pred cells = {int(pred.sum())} / valid {int(valid_bins.sum())}")

    lat_grid, lon_grid = _build_grid_centers(CEARA_BBOX, GRID_HW)
    centroids, sizes = _components_centroids(pred, lat_grid, lon_grid)
    print(f"componentes: {centroids.shape[0]} (tamanhos: {sizes})")
    for i, (lat, lon) in enumerate(centroids[:10]):
        print(f"  comp {i}: lat={lat:.3f} lon={lon:.3f}")

    if centroids.shape[0] and len(df_day):
        D = _haversine_like_km(
            df_day["lat"].to_numpy(),
            df_day["lon"].to_numpy(),
            centroids[:, 0],
            centroids[:, 1],
        )
        d_nearest = D.min(axis=1)
        print(f"\ndistâncias (km) foco→componente mais próxima:")
        print(f"  min={d_nearest.min():.2f} mediana={np.median(d_nearest):.2f} max={d_nearest.max():.2f}")
        print(f"  ≤ 3km: {int(np.sum(d_nearest <= 3))} / {len(df_day)}")
        print(f"  ≤ 5km: {int(np.sum(d_nearest <= 5))} / {len(df_day)}")
        print(f"  ≤ 8km: {int(np.sum(d_nearest <= 8))} / {len(df_day)}")
        print(f"  ≤ 20km: {int(np.sum(d_nearest <= 20))} / {len(df_day)}")
        print(f"  ≤ 50km: {int(np.sum(d_nearest <= 50))} / {len(df_day)}")


if __name__ == "__main__":
    main()
