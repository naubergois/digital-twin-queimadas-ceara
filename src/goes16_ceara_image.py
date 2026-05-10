"""
Descarrega GOES-16 (ABI-L2-CMIPF) via NOAA Open Data em AWS S3 e gera PNG do Ceará.

Dados: bucket público ``noaa-goes16`` — sem chave API; acesso é S3 compatível
(ver https://registry.opendata.aws/noaa-goes/ ).

Fluxo: ``download_cmipf_channel`` → NetCDF local → grade lat/lon (proj. geostacionária)
→ recorte ao ``CEARA_BBOX`` → PNG em ``data/goes16_ceara_png/`` (por defeito).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr

try:
    from pyproj import Proj
except ImportError as e:  # pragma: no cover
    raise ImportError("Instale pyproj: pip install pyproj") from e

# Raiz do repositório (pai de src/)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_writable_mpl_config() -> None:
    """Evita falha quando ~/.matplotlib não é gravável (sandbox, CI)."""
    cfg = _REPO_ROOT / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg))


def _matplotlib_pyplot():
    _ensure_writable_mpl_config()
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as e:  # pragma: no cover
        raise ImportError("Instale matplotlib: pip install matplotlib") from e
    return plt, Normalize


if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.ceara_config import CEARA_BBOX  # noqa: E402

from src.goes16_download import (  # noqa: E402
    GOES16DownloadConfig,
    download_cmipf_channel,
)


def goes_abi_lat_lon(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Converte coordenadas fixas ABI (x, y em radianos) para lat/lon WGS84."""
    proj = ds["goes_imager_projection"]
    h = float(proj.attrs["perspective_point_height"])
    lon0 = float(proj.attrs["longitude_of_projection_origin"])
    sweep = proj.attrs.get("sweep_angle_axis", "x")
    major = float(proj.attrs["semi_major_axis"])
    minor = float(proj.attrs["semi_minor_axis"])

    x = ds["x"].values * h
    y = ds["y"].values * h
    xx, yy = np.meshgrid(x, y)

    p = Proj(proj="geos", h=h, lon_0=lon0, a=major, b=minor, units="m", sweep=sweep)
    lon, lat = p(xx, yy, inverse=True)
    return lat.astype(np.float64), lon.astype(np.float64)


def bbox_slice(lat: np.ndarray, lon: np.ndarray, bbox: dict) -> Tuple[slice, slice]:
    """Índices mínimos que cobrem o bbox (aproximação por retângulo em índices)."""
    mask = (
        (lat >= bbox["min_lat"])
        & (lat <= bbox["max_lat"])
        & (lon >= bbox["min_lon"])
        & (lon <= bbox["max_lon"])
        & np.isfinite(lat)
        & np.isfinite(lon)
    )
    if not mask.any():
        raise ValueError("Nenhum pixel da grade cai dentro do bbox indicado.")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def netcdf_to_ceara_png(
    nc_path: Path,
    png_path: Path,
    *,
    bbox: Optional[dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 120,
    title: Optional[str] = None,
) -> Path:
    """
    Lê um CMIPF NetCDF local e grava PNG recortado ao Ceará.

    ``CMI`` é temperatura de brilho (K) nas bandas térmicas típicas (ex. canal 13).
    """
    bbox = bbox or CEARA_BBOX
    ds = xr.open_dataset(nc_path)
    try:
        lat, lon = goes_abi_lat_lon(ds)
        sl_r, sl_c = bbox_slice(lat, lon, bbox)
        lat_s = lat[sl_r, sl_c]
        lon_s = lon[sl_r, sl_c]
        cmi = ds["CMI"].values[sl_r, sl_c].astype(np.float64)

        valid = np.isfinite(cmi) & np.isfinite(lat_s) & np.isfinite(lon_s)
        # Valores fora de faixa física usual de BT em IR limpo (~180–330 K)
        valid &= (cmi >= 180.0) & (cmi <= 330.0)

        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)

        data = np.ma.masked_where(~valid, cmi)
        if vmin is None:
            vmin = float(np.percentile(cmi[valid], 2)) if valid.any() else 200.0
        if vmax is None:
            vmax = float(np.percentile(cmi[valid], 98)) if valid.any() else 300.0
        plt, Normalize = _matplotlib_pyplot()
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        fig, ax = plt.subplots(figsize=(9, 8), dpi=dpi)
        mesh = ax.pcolormesh(
            lon_s,
            lat_s,
            data,
            cmap="inferno",
            shading="auto",
            norm=norm,
        )
        plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="Temperatura de brilho (K)")
        ax.set_xlim(bbox["min_lon"], bbox["max_lon"])
        ax.set_ylim(bbox["min_lat"], bbox["max_lat"])
        ax.set_xlabel("Longitude °")
        ax.set_ylabel("Latitude °")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        return png_path.resolve()
    finally:
        ds.close()


def fetch_and_save_ceara_png(
    when: datetime,
    *,
    channel: int = 13,
    bbox: Optional[dict] = None,
    raw_dir: Optional[Path] = None,
    png_dir: Optional[Path] = None,
    download_cfg: Optional[GOES16DownloadConfig] = None,
    overwrite_netcdf: bool = False,
    show_progress: bool = True,
) -> Tuple[Path, Path]:
    """
    Descarrega CMIPF para ``when`` (UTC), recorta ao Ceará e grava PNG.

    Devolve ``(caminho_nc, caminho_png)``.
    """
    bbox = bbox or CEARA_BBOX
    download_cfg = download_cfg or GOES16DownloadConfig()
    raw_dir = Path(raw_dir) if raw_dir else download_cfg.cache_dir
    png_dir = Path(png_dir) if png_dir else (_REPO_ROOT / "data" / "goes16_ceara_png")

    nc_path = download_cmipf_channel(
        when,
        channel,
        cfg=download_cfg,
        dest_dir=raw_dir,
        overwrite=overwrite_netcdf,
        show_progress=show_progress,
    )

    when = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    stem = nc_path.stem
    png_path = png_dir / f"{stem}_ceara_ch{channel}.png"

    title = f"GOES-16 ABI CMIPF · Canal {channel} · {when.isoformat()} UTC"
    netcdf_to_ceara_png(nc_path, png_path, bbox=bbox, title=title)
    return nc_path.resolve(), png_path.resolve()


def _parse_when(s: str) -> datetime:
    t = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(
        description="Baixa GOES-16 (S3 NOAA Open Data) e grava PNG do Ceará"
    )
    p.add_argument("--when", required=True, help="Instante UTC ISO, ex.: 2024-10-31T18:05:00Z")
    p.add_argument("--channel", type=int, default=13, help="Canal CMIPF (padrão 13)")
    p.add_argument("--raw-dir", type=Path, default=None, help="Pasta dos NetCDF")
    p.add_argument("--png-dir", type=Path, default=None, help="Pasta de saída PNG")
    p.add_argument("--overwrite-netcdf", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    when = _parse_when(args.when)
    nc, png = fetch_and_save_ceara_png(
        when,
        channel=args.channel,
        raw_dir=args.raw_dir,
        png_dir=args.png_dir,
        overwrite_netcdf=args.overwrite_netcdf,
        show_progress=not args.no_progress,
    )
    print(f"NetCDF: {nc}")
    print(f"PNG:    {png}")


if __name__ == "__main__":
    main()
