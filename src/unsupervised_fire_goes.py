"""
Detecção **não supervisionada** de anomalias térmicas (GOES-16 ABI CMIPF) com
**máscara DQF**, **vários canais** (7 / 13 / 14) e **média ao longo de várias horas UTC**,
comparando com focos **INPE** na mesma grade.

Ver docstring original para referências a *digital twin* e métodos base.

Melhorias operacionais:

- **DQF**: mantém pixels com ``DQF == 0`` (qualidade nominal CMIP).
- **Multi-banda**: ``BT7 - BT14`` (contraste espectral típico de focos) como filtro
  adicional não supervisionado (percentil na cena).
- **Multi-hora**: fusão por **máximo** nas grades por canal (realça picos térmicos breves);
  o **gêmeo digital** em ``goes_fire_digital_twin`` assimila scores horários com persistência.
- **Avaliação**: opcional **dilatação** da máscara INPE (``--truth-dilate``) para tolerância
  geográfica; use ``0`` para comparação estrita célula-a-célula.
- **Calibração opcional** de ``contamination``: procura que maximiza F1 **usando o INPE
  do mesmo dia** — útil para relatório, mas é *violação* de pureza não supervisão para
  implantação operacional (ver ``--calibrate-contamination``).

Gera **métricas** (precisão, recall, F1, IoU, acurácia na grade) e **mapas** TP/FP/FN/TN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, median_filter
from scipy.stats import binned_statistic_2d
from sklearn.ensemble import IsolationForest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.ceara_config import CEARA_BBOX  # noqa: E402
from src.goes16_ceara_image import bbox_slice, goes_abi_lat_lon  # noqa: E402
from src.goes16_download import GOES16DownloadConfig, download_cmipf_channel  # noqa: E402
from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig  # noqa: E402
from src.goes_fire_method_v2 import (  # noqa: E402
    CombinedPersistenceConfig,
    combined_persistence_precision_preset,
    detect_combined_persistence,
)

MethodName = Literal[
    "spatial_residual",
    "isolation_forest",
    "digital_twin",
    "combined_persistence",
    "both",
    "all",
]


def _datetime_column(df: pd.DataFrame) -> pd.Series:
    if "data_pas" in df.columns and df["data_pas"].notna().any():
        return pd.to_datetime(df["data_pas"], utc=True, errors="coerce")
    return pd.to_datetime(df["data_hora_gmt"], utc=True, errors="coerce")


def load_inpe_focos(path: Path, estado_id: int = 23) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = _datetime_column(df)
    df = df.dropna(subset=["datetime", "lat", "lon"])
    eid = pd.to_numeric(df["estado_id"], errors="coerce")
    ce_nome = df["estado"].astype(str).str.upper().str.startswith("CEAR")
    df = df.loc[(eid == float(estado_id)) | (eid.isna() & ce_nome)]
    return df.reset_index(drop=True)


def load_goes_bt_crop(
    nc_path: Path,
    bbox: Optional[dict] = None,
    *,
    dqf_good_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temperatura de brilho (K), lat/lon no recorte; ``valid`` inclui faixa física e DQF."""
    bbox = bbox or CEARA_BBOX
    ds = xr.open_dataset(nc_path)
    try:
        lat_full, lon_full = goes_abi_lat_lon(ds)
        sl_r, sl_c = bbox_slice(lat_full, lon_full, bbox)
        lat = lat_full[sl_r, sl_c]
        lon = lon_full[sl_r, sl_c]
        bt = ds["CMI"].values[sl_r, sl_c].astype(np.float64)
        valid = (
            np.isfinite(bt)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (bt >= 180.0)
            & (bt <= 330.0)
        )
        if dqf_good_only and "DQF" in ds:
            dqf = ds["DQF"].values[sl_r, sl_c]
            valid &= np.isfinite(dqf) & (dqf == 0)
        return bt, lat, lon, valid
    finally:
        ds.close()


def build_lat_lon_edges(bbox: dict, grid_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = grid_hw
    lat_edges = np.linspace(bbox["min_lat"], bbox["max_lat"], h + 1)
    lon_edges = np.linspace(bbox["min_lon"], bbox["max_lon"], w + 1)
    return lat_edges, lon_edges


def bin_mean_bt(
    lat: np.ndarray,
    lon: np.ndarray,
    bt: np.ndarray,
    valid: np.ndarray,
    bbox: dict,
    grid_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    lat_edges, lon_edges = build_lat_lon_edges(bbox, grid_hw)
    mean_bt, _, _, _ = binned_statistic_2d(
        lon[valid],
        lat[valid],
        bt[valid],
        statistic="mean",
        bins=[lon_edges, lat_edges],
    )
    counts, _, _, _ = binned_statistic_2d(
        lon[valid],
        lat[valid],
        bt[valid],
        statistic="count",
        bins=[lon_edges, lat_edges],
    )
    mean_bt = mean_bt.T
    counts = counts.T
    mean_bt[counts == 0] = np.nan
    return mean_bt, counts


def truth_presence_grid(
    df_day: pd.DataFrame,
    bbox: dict,
    grid_hw: Tuple[int, int],
) -> np.ndarray:
    h, w = grid_hw
    lat_edges, lon_edges = build_lat_lon_edges(bbox, grid_hw)
    if df_day.empty:
        return np.zeros((h, w), dtype=bool)
    sums, _, _, _ = binned_statistic_2d(
        df_day["lon"].values,
        df_day["lat"].values,
        np.ones(len(df_day)),
        statistic="sum",
        bins=[lon_edges, lat_edges],
    )
    return (sums.T > 0)


def _filled_bt(bt_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    filled = bt_grid.copy()
    med = np.nanmedian(filled)
    filled = np.nan_to_num(filled, nan=med)
    return filled, med


def detect_spatial_residual(
    bt_grid: np.ndarray,
    *,
    contamination: float = 0.02,
    median_size: int = 9,
) -> np.ndarray:
    filled, _ = _filled_bt(bt_grid)
    bg = median_filter(filled, size=median_size, mode="nearest")
    res = filled - bg
    valid = np.isfinite(bt_grid) & (bt_grid > 0)
    if not valid.any():
        return np.zeros_like(filled, dtype=bool)
    thr = np.percentile(res[valid], 100 * (1 - contamination))
    return (res >= thr) & valid


def detect_spatial_residual_multiband(
    bt13: np.ndarray,
    bt7: Optional[np.ndarray],
    bt14: Optional[np.ndarray],
    *,
    contamination: float = 0.02,
    median_size: int = 9,
    dbt_percentile: float = 85.0,
) -> np.ndarray:
    """Residual em 13 + ``BT7-BT14`` alto (percentil da cena), quando há bandas 7 e 14."""
    pred = detect_spatial_residual(bt13, contamination=contamination, median_size=median_size)
    if bt7 is None or bt14 is None:
        return pred
    dbt = bt7 - bt14
    finite = np.isfinite(dbt) & np.isfinite(bt13)
    if not finite.any():
        return pred
    dthr = np.percentile(dbt[finite], dbt_percentile)
    return pred & finite & (dbt >= dthr)


def detect_isolation_forest(
    bt_grid: np.ndarray,
    *,
    contamination: float = 0.02,
    median_size: int = 9,
    random_state: int = 42,
) -> np.ndarray:
    filled, _ = _filled_bt(bt_grid)
    bg = median_filter(filled, size=median_size, mode="nearest")
    res = filled - bg
    valid = np.isfinite(bt_grid) & (bt_grid > 0)
    h, w = filled.shape
    X = np.column_stack([filled.ravel(), bg.ravel(), res.ravel()])
    m = valid.ravel()
    if m.sum() < 50:
        return np.zeros((h, w), dtype=bool)
    clf = IsolationForest(
        contamination=min(max(contamination, 0.001), 0.5),
        random_state=random_state,
        n_estimators=200,
    )
    labels = np.ones(X.shape[0], dtype=np.int8)
    labels[m] = clf.fit_predict(X[m])
    pred = labels.reshape(h, w) == -1
    return pred & valid


def detect_isolation_forest_multiband(
    bt7: Optional[np.ndarray],
    bt13: np.ndarray,
    bt14: Optional[np.ndarray],
    *,
    contamination: float = 0.02,
    median_size: int = 9,
    random_state: int = 42,
) -> np.ndarray:
    """IF em feições multi-banda; recua para uma banda se faltar 7 ou 14."""
    if bt7 is None or bt14 is None or not np.any(np.isfinite(bt7)) or not np.any(np.isfinite(bt14)):
        return detect_isolation_forest(bt13, contamination=contamination, median_size=median_size, random_state=random_state)

    f7, _ = _filled_bt(bt7)
    f13, _ = _filled_bt(bt13)
    f14, _ = _filled_bt(bt14)
    bg13 = median_filter(f13, size=median_size, mode="nearest")
    res13 = f13 - bg13
    bg14 = median_filter(f14, size=median_size, mode="nearest")
    res14 = f14 - bg14
    dbt = f7 - f14
    valid = np.isfinite(bt13) & (bt13 > 0)
    h, w = f13.shape
    X = np.column_stack([f13.ravel(), bg13.ravel(), res13.ravel(), dbt.ravel(), f14.ravel(), res14.ravel()])
    m = valid.ravel()
    if m.sum() < 80:
        return detect_isolation_forest(bt13, contamination=contamination, median_size=median_size, random_state=random_state)
    clf = IsolationForest(
        contamination=min(max(contamination, 0.001), 0.5),
        random_state=random_state,
        n_estimators=280,
    )
    labels = np.ones(X.shape[0], dtype=np.int8)
    labels[m] = clf.fit_predict(X[m])
    pred = labels.reshape(h, w) == -1
    return pred & valid


def dilate_truth_grid(truth: np.ndarray, iterations: int, *, square_size: int = 3) -> np.ndarray:
    """Expande células INPE para tolerância espacial na avaliação (vizinhança ``square_size``)."""
    if iterations <= 0:
        return truth
    struct = np.ones((square_size, square_size), dtype=bool)
    m = truth.astype(bool)
    for _ in range(iterations):
        m = binary_dilation(m, structure=struct)
    return m


def f_beta_score(precision: float, recall: float, beta: float) -> float:
    """F-measure generalizada; β=1 é F1; β<1 favorece precisão."""
    b = float(max(beta, 1e-9))
    b2 = b * b
    d = b2 * precision + recall
    if d <= 1e-15:
        return 0.0
    return float((1.0 + b2) * precision * recall / d)


def confusion(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    p = pred & valid
    t = truth & valid
    tp = float(np.sum(p & t))
    fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t))
    tn = float(np.sum(~p & ~t))
    denom_pr = tp + fp
    denom_re = tp + fn
    prec = tp / denom_pr if denom_pr else 0.0
    rec = tp / denom_re if denom_re else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    union = np.sum(p | t)
    iou = tp / union if union else 0.0
    tot = tp + tn + fp + fn
    acc = (tp + tn) / tot if tot else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(acc),
        "valid_cells": float(valid.sum()),
    }


def _ensure_mpl():
    cfg = _REPO_ROOT / ".mplconfig"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cfg))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    return plt, BoundaryNorm, ListedColormap, Patch


def plot_real_vs_predicted_map(
    pred: np.ndarray,
    truth: np.ndarray,
    valid_bins: np.ndarray,
    bbox: dict,
    grid_hw: Tuple[int, int],
    out_path: Path,
    *,
    title: str,
    subtitle_metrics: Optional[str] = None,
    bg_bt13: Optional[np.ndarray] = None,
) -> Path:
    """
    Mapa de comparação: cinza = TN, vermelho escuro = FN, laranja = FP, verde = TP.
    Células sem dados ABI ficam por cima do fundo (T_B13 médio se ``bg_bt13`` existir).
    """
    plt, BoundaryNorm, ListedColormap, Patch = _ensure_mpl()

    h, w = grid_hw
    lat_edges, lon_edges = build_lat_lon_edges(bbox, grid_hw)

    layer = np.full((h, w), np.nan, dtype=np.float64)
    tn = (~pred) & (~truth) & valid_bins
    fn = (~pred) & truth & valid_bins
    fp = pred & (~truth) & valid_bins
    tp = pred & truth & valid_bins
    layer[tn] = 0.0
    layer[fn] = 1.0
    layer[fp] = 2.0
    layer[tp] = 3.0

    colors = ["#e8e8e8", "#a50f15", "#fd8d3c", "#238b45"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=140)
    if bg_bt13 is not None and np.isfinite(bg_bt13).any():
        bgm = np.ma.masked_where(~np.isfinite(bg_bt13), bg_bt13)
        ax.pcolormesh(
            lon_edges,
            lat_edges,
            bgm,
            cmap="inferno",
            alpha=0.35,
            shading="auto",
        )
    ax.pcolormesh(lon_edges, lat_edges, layer, cmap=cmap, norm=norm, shading="auto", alpha=0.92)
    ax.set_xlim(bbox["min_lon"], bbox["max_lon"])
    ax.set_ylim(bbox["min_lat"], bbox["max_lat"])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude °")
    ax.set_ylabel("Latitude °")
    ax.set_title(title)
    if subtitle_metrics:
        ax.text(
            0.02,
            0.02,
            subtitle_metrics,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
    legend_elems = [
        Patch(facecolor=colors[3], edgecolor="k", label="TP (acerto)"),
        Patch(facecolor=colors[2], edgecolor="k", label="FP (alarme falso)"),
        Patch(facecolor=colors[1], edgecolor="k", label="FN (fogo não detetado)"),
        Patch(facecolor=colors[0], edgecolor="k", label="TN (sem evento)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path.resolve()


def _calibration_is_better(
    m_new: Dict[str, float],
    pred_new: np.ndarray,
    m_best: Dict[str, float],
    pred_best: np.ndarray,
    *,
    calibrate_beta: float,
) -> bool:
    """
    Compara candidatos na grelha de ``contamination``: maximiza Fβ;
    em empate favorece precisão, IoU, TP, menos FP e máscara mais esparsa.
    """
    eps = 1e-12
    sn = f_beta_score(m_new["precision"], m_new["recall"], calibrate_beta)
    sb = f_beta_score(m_best["precision"], m_best["recall"], calibrate_beta)
    if sn > sb + eps:
        return True
    if sn + eps < sb:
        return False
    if m_new["precision"] > m_best["precision"] + eps:
        return True
    if m_new["precision"] + eps < m_best["precision"]:
        return False
    if m_new["iou"] > m_best["iou"] + eps:
        return True
    if m_new["iou"] + eps < m_best["iou"]:
        return False
    if m_new["tp"] > m_best["tp"] + eps:
        return True
    if m_new["tp"] + eps < m_best["tp"]:
        return False
    if m_new["fp"] + eps < m_best["fp"]:
        return True
    if m_new["fp"] > m_best["fp"] + eps:
        return False
    return int(np.sum(pred_new)) < int(np.sum(pred_best))


def calibrate_contamination(
    truth: np.ndarray,
    valid_bins: np.ndarray,
    bt7: Optional[np.ndarray],
    bt13: np.ndarray,
    bt14: Optional[np.ndarray],
    method: Literal["spatial_residual", "isolation_forest", "digital_twin", "combined_persistence"],
    *,
    hourly_slots: Optional[List[Dict[int, np.ndarray]]] = None,
    twin_cfg: Optional[GOESFireDigitalTwinConfig] = None,
    combined_persistence_cfg: Optional[CombinedPersistenceConfig] = None,
    calibrate_beta: float = 1.0,
    n_steps: int = 32,
) -> Tuple[float, np.ndarray]:
    """Escolhe ``contamination`` que maximiza Fβ (β=1 → F1; β<1 privilegia precisão — usa INPE)."""
    best_pred = np.zeros_like(valid_bins, dtype=bool)
    best_m: Optional[Dict[str, float]] = None
    best_c = 0.03
    beta = float(max(calibrate_beta, 1e-6))
    for c in np.linspace(0.004, 0.18, n_steps):
        if method == "spatial_residual":
            pred = detect_spatial_residual_multiband(bt13, bt7, bt14, contamination=float(c))
        elif method == "isolation_forest":
            pred = detect_isolation_forest_multiband(bt7, bt13, bt14, contamination=float(c))
        elif method == "combined_persistence":
            pred = (
                detect_combined_persistence(
                    hourly_slots,
                    valid_bins,
                    float(c),
                    cfg=combined_persistence_cfg,
                )
                if hourly_slots
                else np.zeros_like(valid_bins, dtype=bool)
            )
        else:
            if not hourly_slots:
                pred = np.zeros_like(valid_bins, dtype=bool)
            else:
                twin = GOESFireDigitalTwin(bt13.shape, twin_cfg or GOESFireDigitalTwinConfig())
                twin.ingest_series(hourly_slots)
                pred = twin.predict_mask(float(c), valid_bins)
        m = confusion(pred, truth, valid_bins)
        if best_m is None or _calibration_is_better(m, pred, best_m, best_pred, calibrate_beta=beta):
            best_m = m
            best_c = float(c)
            best_pred = pred
    assert best_m is not None
    return best_c, best_pred


@dataclass
class DayEvalResult:
    day_utc: str
    goes_nc: str
    method: str
    metrics: Dict[str, Any]
    contamination_used: float = 1.0
    channels_used: List[int] = field(default_factory=list)
    hours_utc_used: List[int] = field(default_factory=list)


def find_local_goes_nc(raw_dir: Path, day_utc: date, channel: int = 13) -> Path:
    raw_dir = Path(raw_dir)
    doy = datetime(day_utc.year, day_utc.month, day_utc.day).timetuple().tm_yday
    tag = f"_s{day_utc.year}{doy:03d}"
    tag_ch = f"M6C{channel:02d}"
    for p in sorted(raw_dir.glob("*.nc")):
        if tag in p.name and tag_ch in p.name:
            return p
    raise FileNotFoundError(
        f"Nenhum NetCDF em {raw_dir} com '{tag}' e canal '{tag_ch}' (--skip-download)."
    )


def ensure_goes_netcdf(
    when: datetime,
    channel: int,
    raw_dir: Path,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    cfg = GOES16DownloadConfig(cache_dir=raw_dir)
    return download_cmipf_channel(
        when,
        channel,
        cfg=cfg,
        dest_dir=raw_dir,
        overwrite=overwrite,
        show_progress=show_progress,
    )


def collect_hourly_band_grids(
    day_utc: date,
    hours_utc: Sequence[int],
    channels: Sequence[int],
    bbox: dict,
    grid_hw: Tuple[int, int],
    raw_dir: Path,
    *,
    skip_download: bool,
    overwrite: bool,
    use_dqf: bool,
    show_progress: bool,
) -> Tuple[List[Dict[int, np.ndarray]], List[str]]:
    """Lista de snapshots por hora: cada entrada é ``{canal: grade (H,W)}``."""
    hourly: List[Dict[int, np.ndarray]] = []
    nc_refs: List[str] = []

    if skip_download and len(hours_utc) > 1:
        warnings.warn(
            "Com --skip-download só é garantido um granulo por dia/canal; "
            "use uma hora ou ficheiros locais para todas as horas.",
            stacklevel=2,
        )

    for hour in hours_utc:
        when = datetime(day_utc.year, day_utc.month, day_utc.day, int(hour), tzinfo=timezone.utc)
        slot: Dict[int, np.ndarray] = {}
        for ch in channels:
            if skip_download:
                path = find_local_goes_nc(raw_dir, day_utc, ch)
            else:
                path = ensure_goes_netcdf(when, ch, raw_dir, overwrite=overwrite, show_progress=show_progress)
            nc_refs.append(str(path))
            bt, lat, lon, valid = load_goes_bt_crop(path, bbox=bbox, dqf_good_only=use_dqf)
            mean_bt, _ = bin_mean_bt(lat, lon, bt, valid, bbox, grid_hw)
            slot[ch] = mean_bt
        hourly.append(slot)

    return hourly, nc_refs


def merge_band_grids_max(hourly: List[Dict[int, np.ndarray]], channels: Sequence[int]) -> Dict[int, np.ndarray]:
    """Fusão multi-hora por **máximo** (realça picos térmicos breves)."""
    out: Dict[int, np.ndarray] = {}
    for ch in channels:
        out[ch] = np.nanmax(np.stack([slot[ch] for slot in hourly], axis=0), axis=0)
    return out


def intersect_valid_bins_hourly(
    hourly: List[Dict[int, np.ndarray]],
    channels: Sequence[int],
) -> np.ndarray:
    """Intersecção das máscaras ``fin(...)`` em todas as horas."""
    if not hourly:
        raise ValueError("hourly vazio")
    v = np.ones(hourly[0][13].shape, dtype=bool)
    for slot in hourly:
        vi = np.isfinite(slot[13])
        for ch in channels:
            vi &= np.isfinite(slot[ch])
        v &= vi
    return v


def evaluate_one_day_enhanced(
    hourly_grids: List[Dict[int, np.ndarray]],
    band_grids: Dict[int, np.ndarray],
    df_inpe: pd.DataFrame,
    day_utc: date,
    *,
    grid_hw: Tuple[int, int],
    bbox: Optional[dict] = None,
    contamination: float = 0.02,
    method: MethodName = "both",
    calibrate_contamination_flag: bool = False,
    truth_dilate_iters: int = 0,
    twin_cfg: Optional[GOESFireDigitalTwinConfig] = None,
    combined_persistence_cfg: Optional[CombinedPersistenceConfig] = None,
    calibrate_beta: float = 1.0,
) -> List[Tuple[DayEvalResult, np.ndarray]]:
    bbox = bbox or CEARA_BBOX
    bt13 = band_grids.get(13)
    if bt13 is None:
        raise ValueError("Canal 13 é obrigatório nas grades agregadas.")
    bt7 = band_grids.get(7)
    bt14 = band_grids.get(14)

    ch_keys = sorted(band_grids.keys())
    valid_bins = intersect_valid_bins_hourly(hourly_grids, ch_keys)

    d0 = datetime(day_utc.year, day_utc.month, day_utc.day, tzinfo=timezone.utc)
    d1 = d0 + pd.Timedelta(days=1)
    day_mask = (df_inpe["datetime"] >= d0) & (df_inpe["datetime"] < d1)
    df_day = df_inpe.loc[day_mask]
    truth_raw = truth_presence_grid(df_day, bbox, grid_hw)
    truth_eval = dilate_truth_grid(truth_raw, truth_dilate_iters)

    if method == "both":
        methods: List[str] = ["spatial_residual", "isolation_forest"]
    elif method == "all":
        methods = [
            "spatial_residual",
            "isolation_forest",
            "digital_twin",
            "combined_persistence",
        ]
    else:
        methods = [method]

    outs: List[Tuple[DayEvalResult, np.ndarray]] = []
    for name in methods:
        c_use = contamination
        if calibrate_contamination_flag:
            if name == "digital_twin":
                c_use, pred = calibrate_contamination(
                    truth_eval,
                    valid_bins,
                    bt7,
                    bt13,
                    bt14,
                    "digital_twin",
                    hourly_slots=hourly_grids,
                    twin_cfg=twin_cfg,
                    calibrate_beta=calibrate_beta,
                )
            elif name == "combined_persistence":
                c_use, pred = calibrate_contamination(
                    truth_eval,
                    valid_bins,
                    bt7,
                    bt13,
                    bt14,
                    "combined_persistence",
                    hourly_slots=hourly_grids,
                    twin_cfg=twin_cfg,
                    combined_persistence_cfg=combined_persistence_cfg,
                    calibrate_beta=calibrate_beta,
                )
            else:
                c_use, pred = calibrate_contamination(
                    truth_eval,
                    valid_bins,
                    bt7,
                    bt13,
                    bt14,
                    name,  # type: ignore[arg-type]
                    calibrate_beta=calibrate_beta,
                )
        elif name == "spatial_residual":
            pred = detect_spatial_residual_multiband(bt13, bt7, bt14, contamination=c_use)
        elif name == "isolation_forest":
            pred = detect_isolation_forest_multiband(bt7, bt13, bt14, contamination=c_use)
        elif name == "combined_persistence":
            pred = detect_combined_persistence(
                hourly_grids,
                valid_bins,
                c_use,
                cfg=combined_persistence_cfg,
            )
        else:
            twin = GOESFireDigitalTwin(bt13.shape, twin_cfg or GOESFireDigitalTwinConfig())
            twin.ingest_series(hourly_grids)
            pred = twin.predict_mask(c_use, valid_bins)

        met = confusion(pred, truth_eval, valid_bins)
        beta_rep = float(max(calibrate_beta, 1e-6))
        met.update(
            {
                "n_focos_inpe": int(len(df_day)),
                "pred_cells": int(np.sum(pred & valid_bins)),
                "truth_cells": int(np.sum(truth_eval & valid_bins)),
                "truth_cells_raw": int(np.sum(truth_raw & valid_bins)),
                "truth_dilate_iters": int(truth_dilate_iters),
                "calibrated_contamination": bool(calibrate_contamination_flag),
                "calibrate_beta": beta_rep,
                "f_beta": f_beta_score(met["precision"], met["recall"], beta_rep),
            }
        )
        res = DayEvalResult(
            day_utc=d0.strftime("%Y-%m-%d"),
            goes_nc="",
            method=name,
            metrics=met,
            contamination_used=float(c_use),
            channels_used=ch_keys,
            hours_utc_used=[],
        )
        outs.append((res, pred))
    return outs


def run_dates_cli(
    dates: List[date],
    inpe_csv: Path,
    *,
    hours_utc: Sequence[int],
    channels: Sequence[int],
    grid_hw: Tuple[int, int] = (72, 72),
    contamination: float = 0.02,
    method: MethodName = "both",
    raw_dir: Optional[Path] = None,
    output_json: Optional[Path] = None,
    map_dir: Optional[Path] = None,
    skip_download: bool = False,
    overwrite_goes: bool = False,
    use_dqf: bool = True,
    show_progress: bool = True,
    calibrate_contamination_flag: bool = False,
    truth_dilate_iters: int = 1,
    twin_cfg: Optional[GOESFireDigitalTwinConfig] = None,
    combined_persistence_cfg: Optional[CombinedPersistenceConfig] = None,
    calibrate_beta: float = 1.0,
) -> List[DayEvalResult]:
    raw_dir = Path(raw_dir) if raw_dir else _REPO_ROOT / "data" / "goes16_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    map_dir = Path(map_dir) if map_dir else _REPO_ROOT / "data" / "goes16_eval" / "maps"

    df = load_inpe_focos(inpe_csv)
    all_results: List[DayEvalResult] = []

    tc = twin_cfg or GOESFireDigitalTwinConfig()

    for d in dates:
        hourly_grids, nc_refs = collect_hourly_band_grids(
            d,
            hours_utc,
            channels,
            CEARA_BBOX,
            grid_hw,
            raw_dir,
            skip_download=skip_download,
            overwrite=overwrite_goes,
            use_dqf=use_dqf,
            show_progress=show_progress,
        )
        band_grids = merge_band_grids_max(hourly_grids, channels)
        valid_bins = intersect_valid_bins_hourly(hourly_grids, sorted(channels))
        pairs = evaluate_one_day_enhanced(
            hourly_grids,
            band_grids,
            df,
            d,
            grid_hw=grid_hw,
            contamination=contamination,
            method=method,
            calibrate_contamination_flag=calibrate_contamination_flag,
            truth_dilate_iters=truth_dilate_iters,
            twin_cfg=tc,
            combined_persistence_cfg=combined_persistence_cfg,
            calibrate_beta=calibrate_beta,
        )

        d0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        truth_raw = truth_presence_grid(
            df.loc[(df["datetime"] >= d0) & (df["datetime"] < d0 + pd.Timedelta(days=1))],
            CEARA_BBOX,
            grid_hw,
        )
        truth_plot = dilate_truth_grid(truth_raw, truth_dilate_iters)

        for res, pred in pairs:
            res.hours_utc_used = list(hours_utc)
            res.goes_nc = nc_refs[0] if len(nc_refs) == 1 else f"{len(nc_refs)} ficheiros GOES"
            all_results.append(res)

            if map_dir:
                m = res.metrics
                sub = (
                    f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                    f"Fβ={m['f_beta']:.3f} (β={m['calibrate_beta']:.3g}) "
                    f"IoU={m['iou']:.3f} Acc={m['accuracy']:.3f}\n"
                    f"c={res.contamination_used:.4f} | dilate_truth={truth_dilate_iters} | "
                    f"twin_p={tc.persistence:.2f} σ={tc.gaussian_sigma:.2f} "
                    f"fusão={tc.fusion} limiar={tc.threshold_mode} LOF={tc.lof_neighbors}\n"
                    f"DQFs={'sim' if use_dqf else 'não'} | canais={res.channels_used} | "
                    f"horasUTC={res.hours_utc_used}"
                )
                png = map_dir / f"real_vs_previsto_{res.day_utc}_{res.method}.png"
                plot_real_vs_predicted_map(
                    pred,
                    truth_plot,
                    valid_bins,
                    CEARA_BBOX,
                    grid_hw,
                    png,
                    title=f"INPE (real⊕dilate) vs previsto — {res.day_utc} — {res.method}",
                    subtitle_metrics=sub,
                    bg_bt13=band_grids.get(13),
                )

    if output_json:
        payload = []
        for r in all_results:
            row = {
                "day": r.day_utc,
                "method": r.method,
                "goes_nc": r.goes_nc,
                "contamination_used": r.contamination_used,
                "channels": r.channels_used,
                "hours_utc": r.hours_utc_used,
                "truth_dilate_iters": truth_dilate_iters,
                "twin_persistence": tc.persistence,
                "twin_gaussian_sigma": tc.gaussian_sigma,
                "twin_fusion": tc.fusion,
                "twin_threshold_mode": tc.threshold_mode,
                "twin_lof_neighbors": tc.lof_neighbors,
                "twin_multiscale_median_sizes": list(tc.multiscale_median_sizes),
                **r.metrics,
            }
            payload.append(row)
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return all_results


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_float_tuple(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_optional_float_tuple(s: Optional[str]) -> Optional[Tuple[float, ...]]:
    if s is None or not str(s).strip():
        return None
    return _parse_float_tuple(str(s))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="GOES-16 não supervisionado vs INPE — métricas + mapas (DQF, multi-banda, multi-hora)"
    )
    p.add_argument("--inpe-csv", type=Path, required=True)
    p.add_argument("--dates", nargs="+", required=True, help="Dias UTC AAAA-MM-DD")
    p.add_argument("--hours-utc", type=str, default="16,17,18", help="Lista separada por vírgulas")
    p.add_argument("--channels", type=str, default="7,13,14", help="Canais CMIPF, ex.: 7,13,14")
    p.add_argument("--grid", type=int, default=72)
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument(
        "--calibrate-contamination",
        action="store_true",
        help="Ajusta contamination ao INPE do dia (optimiza Fβ; ver --calibrate-beta)",
    )
    p.add_argument(
        "--calibrate-beta",
        type=float,
        default=None,
        metavar="β",
        help="Fβ na calibração: β<1 favorece precisão (ex.: 0.5). Omitido → 1.0 ou 0.5 com --precision-focus",
    )
    p.add_argument(
        "--precision-focus",
        action="store_true",
        help="Usa preset alto-precision em combined_persistence (ignora --cp-*); "
        "com --calibrate-contamination e β omitido usa F0.5",
    )
    p.add_argument(
        "--truth-dilate",
        type=int,
        default=1,
        help="Iterações de dilatação na máscara INPE para tolerância espacial (0=desligado)",
    )
    p.add_argument(
        "--method",
        choices=(
            "spatial_residual",
            "isolation_forest",
            "digital_twin",
            "combined_persistence",
            "both",
            "all",
        ),
        default="all",
        help="both=spatial+IF; all=+twin+combined_persistence (Linha B, doc METODOLOGIA_NOVA_PROPOSTA.md)",
    )
    p.add_argument("--twin-persistence", type=float, default=0.5, help="Memória do twin entre horas [0,1]")
    p.add_argument("--twin-sigma", type=float, default=1.2, help="Suavização Gaussiana no campo de risco")
    p.add_argument("--twin-dbt-weight", type=float, default=0.55, help="Peso do BTD (7−14) no score horário")
    p.add_argument(
        "--twin-multiscale",
        type=str,
        default="5,9,15",
        help="Tamanhos de mediana multi-escala (ímpares), separados por vírgula",
    )
    p.add_argument(
        "--twin-fusion",
        choices=("max_persist", "prob_or"),
        default="prob_or",
        help="Fusão temporal: max_persist ou combinação probabilística prob_or",
    )
    p.add_argument(
        "--twin-threshold",
        choices=("percentile", "gmm2"),
        default="percentile",
        help="percentil (robusto em cenas curtas); gmm2 = posterior do modo quente (GMM-2)",
    )
    p.add_argument(
        "--twin-lof-neighbors",
        type=int,
        default=24,
        help="Vizinhos LOF para refinamento espacial (0 desliga)",
    )
    p.add_argument(
        "--cp-hour-pct",
        type=float,
        default=88.0,
        help="Percentil base por hora para activação em s_t (combined_persistence)",
    )
    p.add_argument(
        "--cp-hour-pcts",
        type=str,
        default=None,
        help="Percentis por hora, ex.: 84,86,88 (sobrepõe --cp-hour-pct; len = nº horas)",
    )
    p.add_argument(
        "--cp-no-adaptive-pct",
        action="store_true",
        help="Desliga ajuste do percentil quando a cena horária tem pouco contraste",
    )
    p.add_argument(
        "--cp-weights",
        type=str,
        default="0.42,0.21,0.37",
        help="Pesos pico,média,persistência (normalizados internamente)",
    )
    p.add_argument("--cp-persist-scale", type=float, default=0.56, help="Numerador em persist_scale/n_eff")
    p.add_argument("--cp-persist-floor", type=float, default=0.16, help="Chão da fração de persistência")
    p.add_argument(
        "--cp-min-persist",
        type=float,
        default=None,
        help="Fração mínima fixa (substitui escala/chão); omitido = automático",
    )
    p.add_argument(
        "--cp-min-hours",
        type=int,
        default=-1,
        help="Mín. horas activas (-1=auto, 0=só fração, ≥1 fixo)",
    )
    p.add_argument(
        "--cp-weak-open-after",
        type=int,
        default=260,
        help="Opening fraco (cruz) só se pred_cells ≥ este valor; 0=desliga",
    )
    p.add_argument("--cp-weak-iters", type=int, default=1, help="Iterações do opening fraco")
    p.add_argument(
        "--cp-weak-struct",
        choices=("cross", "box"),
        default="cross",
        help="cross=4-vizinhos (mais fraco); box=3×3",
    )
    p.add_argument(
        "--cp-min-cc",
        type=int,
        default=3,
        help="Remove componentes com menos células (≤1 desliga)",
    )
    p.add_argument("--cp-morph-iters", type=int, default=0, help="Opening 3×3 extra (agressivo)")
    p.add_argument(
        "--cp-multiscale",
        type=str,
        default="5,9,15",
        help="Medianas multi-escala no score horário (combined_persistence)",
    )
    p.add_argument(
        "--cp-dbt-weight",
        type=float,
        default=0.55,
        help="Peso BTD (7−14) dentro do score horário quando multi-banda",
    )
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--map-dir",
        type=Path,
        default=None,
        help="Pasta para PNG real vs previsto (defeito: data/goes16_eval/maps)",
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--overwrite-goes", action="store_true")
    p.add_argument("--no-dqf", action="store_true", help="Não filtrar por DQF==0")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    dates = [datetime.strptime(s, "%Y-%m-%d").date() for s in args.dates]
    hours = _parse_int_list(args.hours_utc)
    channels = _parse_int_list(args.channels)
    if 13 not in channels:
        channels.append(13)
        channels = sorted(set(channels))

    twin_cfg = GOESFireDigitalTwinConfig(
        persistence=args.twin_persistence,
        gaussian_sigma=args.twin_sigma,
        dbt_weight=args.twin_dbt_weight,
        multiscale_median_sizes=_parse_int_tuple(args.twin_multiscale),
        fusion=args.twin_fusion,
        threshold_mode=args.twin_threshold,
        lof_neighbors=max(0, args.twin_lof_neighbors),
    )

    if args.precision_focus:
        combined_persistence_cfg = combined_persistence_precision_preset()
    else:
        combined_persistence_cfg = CombinedPersistenceConfig(
            hour_active_percentile=float(args.cp_hour_pct),
            hour_active_percentiles=_parse_optional_float_tuple(args.cp_hour_pcts),
            adaptive_hour_percentile=not args.cp_no_adaptive_pct,
            weights_peak_mean_persist=_parse_float_tuple(args.cp_weights),
            persist_scale=float(args.cp_persist_scale),
            persist_floor=float(args.cp_persist_floor),
            min_persist_frac=args.cp_min_persist,
            min_active_hours=int(args.cp_min_hours),
            weak_open_after_pred_cells=int(args.cp_weak_open_after),
            weak_open_iterations=max(0, int(args.cp_weak_iters)),
            weak_open_connectivity=args.cp_weak_struct,
            min_component_cells=int(args.cp_min_cc),
            morph_open_iterations=max(0, int(args.cp_morph_iters)),
            median_sizes=_parse_int_tuple(args.cp_multiscale),
            dbt_weight=float(args.cp_dbt_weight),
        )

    calibrate_beta = args.calibrate_beta
    if calibrate_beta is None:
        calibrate_beta = 0.5 if args.precision_focus else 1.0
    calibrate_beta = float(max(calibrate_beta, 1e-6))

    results = run_dates_cli(
        dates,
        args.inpe_csv,
        hours_utc=hours,
        channels=channels,
        grid_hw=(args.grid, args.grid),
        contamination=args.contamination,
        method=args.method,
        raw_dir=args.raw_dir,
        output_json=args.output_json,
        map_dir=args.map_dir,
        skip_download=args.skip_download,
        overwrite_goes=args.overwrite_goes,
        use_dqf=not args.no_dqf,
        show_progress=not args.no_progress,
        calibrate_contamination_flag=args.calibrate_contamination,
        truth_dilate_iters=max(0, args.truth_dilate),
        twin_cfg=twin_cfg,
        combined_persistence_cfg=combined_persistence_cfg,
        calibrate_beta=calibrate_beta,
    )
    for r in results:
        row = {"day": r.day_utc, "method": r.method, **r.metrics}
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
