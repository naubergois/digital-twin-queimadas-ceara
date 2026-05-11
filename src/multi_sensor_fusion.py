"""
Fusão multi-sensor GOES-16 + VIIRS (DTEC §5).

VIIRS-I AF (375 m, S-NPP / NOAA-20 / NOAA-21) é o **selo semântico**
mais fiável para fogo numa região tropical: F1 reportado contra Landsat
0,80–0,92 (cf. ``docs/METODOLOGIA_DTEC_F1_080.md``). Aqui o módulo:

1. **Carrega** detecções VIIRS num DataFrame compatível com NASA FIRMS
   (colunas mínimas: ``lat``, ``lon``, ``datetime`` opcional ``confidence``).
2. **Binariza** para a grade GOES (booleano por célula).
3. **Funde** com a probabilidade do gêmeo digital de várias formas:

   - ``AND``   — só quando GOES e VIIRS concordam (máxima precisão).
   - ``OR``    — união (máxima cobertura).
   - ``GATED`` — predizer GOES, mas só nas vizinhanças de cada detecção
     VIIRS expandidas por ``gate_radius_km``. Compromisso útil.
   - ``WEIGHTED`` — score = ``w_g·P_GOES + w_v·indicador_VIIRS_local``.

Nenhuma destas formas substitui o gêmeo digital — todas o **mantêm como
backbone**. VIIRS entra como sinal adicional de confirmação.

O módulo também sintetiza VIIRS realista a partir de focos de referência
(``synthesize_viirs_proxy``) para demonstrar o pipeline sem chamadas a APIs
externas. **Em produção** ligar `firms_download.py` (ver `scripts/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure

from src.event_centric import (
    KM_PER_DEG_LAT,
    _build_grid_centers,
    _filter_focos,
    _haversine_like_km,
    _km_per_deg_lon,
)
from src.unsupervised_fire_goes import build_lat_lon_edges


FusionMode = Literal["and", "or", "gated", "weighted"]


@dataclass
class FusionConfig:
    mode: FusionMode = "gated"
    gate_radius_km: float = 5.0
    """Raio em km para considerar uma célula GOES "perto" de um foco VIIRS."""
    weight_goes: float = 0.6
    weight_viirs: float = 0.4
    viirs_min_confidence: Optional[float] = None
    """Filtro opcional na confiança VIIRS (0–100 no FIRMS)."""


def load_viirs_firms_csv(path) -> pd.DataFrame:
    """
    Lê um CSV no formato NASA FIRMS VIIRS_NOAA20_NRT / VIIRS_SNPP_NRT.

    Normaliza colunas para ``lat``, ``lon``, ``datetime`` (UTC) e
    ``confidence`` (categórica low/n/h convertida em 30/60/90, ou numérica).
    """
    df = pd.read_csv(path, low_memory=False)
    rename = {}
    if "latitude" in df.columns:
        rename["latitude"] = "lat"
    if "longitude" in df.columns:
        rename["longitude"] = "lon"
    df = df.rename(columns=rename)
    if "acq_date" in df.columns and "acq_time" in df.columns:
        t = df["acq_time"].astype(int).astype(str).str.zfill(4)
        ts = df["acq_date"].astype(str) + " " + t.str[:2] + ":" + t.str[2:4] + ":00"
        df["datetime"] = pd.to_datetime(ts, utc=True, errors="coerce")
    elif "datetime" not in df.columns:
        df["datetime"] = pd.NaT
    if "confidence" in df.columns and df["confidence"].dtype == object:
        m = {"l": 30.0, "n": 60.0, "h": 90.0}
        df["confidence"] = df["confidence"].str.lower().map(m).astype(float)
    return df


def synthesize_viirs_proxy(
    truth_focos: pd.DataFrame,
    *,
    bbox: dict,
    detection_rate: float = 0.8,
    spatial_jitter_km: float = 1.5,
    false_positive_rate: float = 0.01,
    n_cells_in_bbox: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Simula um feed VIIRS realista a partir de focos de referência:

    - Detecta uma fração ``detection_rate`` dos focos verdadeiros.
    - Cada detecção é jitter-deslocada por ~ ``spatial_jitter_km`` km
      (375 m × parallax → ~1–2 km típico).
    - Adiciona ``false_positive_rate × n_cells`` falsos positivos
      uniformemente no bbox (modela detecção em solo nu / borda de nuvem).

    Útil para demonstrar a fusão sem depender de download FIRMS.
    """
    rng = np.random.default_rng(seed)
    if truth_focos.empty:
        return truth_focos.iloc[:0].copy()

    n = int(len(truth_focos))
    keep = rng.random(n) < float(detection_rate)
    base = truth_focos.iloc[np.flatnonzero(keep)].copy()

    # Jitter espacial — converter km → graus aproximadamente
    lat_ref = float(base["lat"].mean())
    kx = _km_per_deg_lon(lat_ref)
    ky = KM_PER_DEG_LAT
    j_lat_km = rng.normal(0.0, spatial_jitter_km / np.sqrt(2.0), size=len(base))
    j_lon_km = rng.normal(0.0, spatial_jitter_km / np.sqrt(2.0), size=len(base))
    base["lat"] = base["lat"].to_numpy() + j_lat_km / ky
    base["lon"] = base["lon"].to_numpy() + j_lon_km / kx

    # Falsos positivos uniformes no bbox
    n_fp = max(0, int(false_positive_rate * float(n_cells_in_bbox)))
    if n_fp:
        fp_lat = rng.uniform(bbox["min_lat"], bbox["max_lat"], size=n_fp)
        fp_lon = rng.uniform(bbox["min_lon"], bbox["max_lon"], size=n_fp)
        fp = pd.DataFrame({
            "lat": fp_lat,
            "lon": fp_lon,
            "datetime": pd.to_datetime(["2024-10-31T17:24:00Z"] * n_fp, utc=True),
        })
        base = pd.concat([base[["lat", "lon", "datetime"]], fp], ignore_index=True)

    # Confiança "VIIRS": real detecção 90, falso positivo 60 (heurístico)
    base = base.reset_index(drop=True)
    base["confidence"] = 90.0
    if n_fp:
        base.loc[base.index[-n_fp:], "confidence"] = 60.0
    base["satellite"] = "VIIRS_proxy"
    base["instrument"] = "VIIRS-I"
    return base


def viirs_cell_mask(
    df_viirs: pd.DataFrame,
    bbox: dict,
    grid_hw: Tuple[int, int],
    *,
    day_utc: Optional[Tuple[datetime, datetime]] = None,
    min_confidence: Optional[float] = None,
) -> np.ndarray:
    """Devolve booleano H×W indicando células com ≥ 1 detecção VIIRS."""
    h, w = grid_hw
    out = np.zeros(grid_hw, dtype=bool)
    if df_viirs is None or df_viirs.empty:
        return out
    df = df_viirs
    if day_utc is not None and "datetime" in df.columns:
        d0, d1 = day_utc
        df = df.loc[
            df["datetime"].notna()
            & (df["datetime"] >= d0)
            & (df["datetime"] < d1)
        ]
    if min_confidence is not None and "confidence" in df.columns:
        df = df.loc[df["confidence"].fillna(0) >= float(min_confidence)]
    df = _filter_focos(df, bbox)
    if df.empty:
        return out
    lat_edges, lon_edges = build_lat_lon_edges(bbox, grid_hw)
    rows = np.digitize(df["lat"].to_numpy(), lat_edges) - 1
    cols = np.digitize(df["lon"].to_numpy(), lon_edges) - 1
    keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[rows[keep], cols[keep]] = True
    return out


def _gate_dilate_cells(mask: np.ndarray, radius_km: float, bbox: dict, grid_hw: Tuple[int, int]) -> np.ndarray:
    """Dilata uma máscara binária por raio em km → número aproximado de células."""
    h, w = grid_hw
    lat_range = bbox["max_lat"] - bbox["min_lat"]
    lon_range = bbox["max_lon"] - bbox["min_lon"]
    cell_lat_km = lat_range * KM_PER_DEG_LAT / h
    lat_ref = 0.5 * (bbox["min_lat"] + bbox["max_lat"])
    cell_lon_km = lon_range * _km_per_deg_lon(lat_ref) / w
    cell_km = float(np.mean([cell_lat_km, cell_lon_km]))
    n_iters = max(1, int(np.round(float(radius_km) / max(cell_km, 1e-6))))
    struct = generate_binary_structure(2, 2)
    d = mask.copy()
    for _ in range(n_iters):
        d = binary_dilation(d, structure=struct)
    return d


@dataclass
class FusionResult:
    pred_mask: np.ndarray
    viirs_mask: np.ndarray
    viirs_dilated: np.ndarray
    mode: FusionMode


def fuse_goes_with_viirs(
    goes_pred: np.ndarray,
    goes_prob: Optional[np.ndarray],
    df_viirs: pd.DataFrame,
    bbox: dict,
    grid_hw: Tuple[int, int],
    valid_bins: np.ndarray,
    *,
    cfg: Optional[FusionConfig] = None,
    day_utc: Optional[Tuple[datetime, datetime]] = None,
) -> FusionResult:
    """
    Funde a máscara/probabilidade do gêmeo digital com detecções VIIRS.

    ``goes_pred`` é o resultado actual do pipeline DTEC (booleano H×W).
    ``goes_prob`` é opcional; só usado em ``mode='weighted'``.
    """
    c = cfg or FusionConfig()
    v_mask = viirs_cell_mask(
        df_viirs, bbox, grid_hw,
        day_utc=day_utc,
        min_confidence=c.viirs_min_confidence,
    )
    v_dil = _gate_dilate_cells(v_mask, c.gate_radius_km, bbox, grid_hw) & valid_bins

    if c.mode == "and":
        pred = goes_pred & v_dil
    elif c.mode == "or":
        pred = (goes_pred | v_dil) & valid_bins
    elif c.mode == "gated":
        pred = (goes_pred & v_dil) | (v_mask & valid_bins)
        # gate: GOES só conta perto de VIIRS, mas VIIRS sozinha já é deteção
    elif c.mode == "weighted":
        if goes_prob is None:
            raise ValueError("mode='weighted' precisa de goes_prob")
        prob_g = np.where(valid_bins, np.clip(goes_prob, 0.0, 1.0), 0.0)
        prob_v = v_dil.astype(np.float64) * 1.0 + v_mask.astype(np.float64) * 0.5
        prob_v = np.clip(prob_v, 0.0, 1.0)
        wsum = max(c.weight_goes + c.weight_viirs, 1e-9)
        score = (c.weight_goes * prob_g + c.weight_viirs * prob_v) / wsum
        # limiar global em top-K de prob_g (mantém esparsidade)
        if goes_pred.any():
            k = int(goes_pred.sum())
            thr = np.partition(score.ravel(), -k)[-k] if k > 0 else 0.5
            pred = (score >= thr) & valid_bins
        else:
            pred = (prob_v >= 0.5) & valid_bins
    else:
        raise ValueError(f"modo de fusão desconhecido: {c.mode!r}")

    return FusionResult(
        pred_mask=pred & valid_bins,
        viirs_mask=v_mask,
        viirs_dilated=v_dil,
        mode=c.mode,
    )
