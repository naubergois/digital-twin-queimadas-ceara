"""
Avaliação **event-centric** de máscaras de previsão GOES contra focos INPE
(ver ``docs/METODOLOGIA_DTEC_F1_080.md``).

Diferença para a avaliação atual (pixel-grid em
``unsupervised_fire_goes.confusion``):

- **Verdade** = cada foco INPE é **um evento** com posição (lat, lon) e tempo.
- **Previsão** = cada **componente conexa** da máscara binária é um evento
  candidato, representado pelo seu centróide em (lat, lon).
- Um par (foco, componente) **casa** se a distância do foco ao centróide
  for ≤ ``radius_km``. Quando há cubo horário (futuro), também se exige
  ``|t_foco − t_evento| ≤ dt_min``.

A métrica continua a chamar-se F1, mas é calculada por **eventos**, não por
células — solução directa para os desalinhamentos de escala e tempo
identificados em ``METODOLOGIA_NOVA_PROPOSTA.md`` §1.

Convenções:

- ``tp_for_recall``  — focos INPE com pelo menos uma componente prevista a ≤ R.
- ``tp_for_precision`` — componentes previstas com pelo menos um foco a ≤ R.
- ``precision = tp_for_precision / (n_componentes)``
- ``recall = tp_for_recall / (n_focos)``
- ``f1 = 2·P·R/(P+R)``

Pode haver assimetria entre ``tp_for_recall`` e ``tp_for_precision`` (uma
componente que cobre vários focos; um foco entre duas componentes próximas).
Isto é intencional e segue a prática em literatura de Active Fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label

KM_PER_DEG_LAT = 111.32

MatchingMode = Literal["centroid", "cell"]
"""
``centroid``: cada componente é representada pelo seu centróide. Bom para
contar **eventos** discretos, mas penaliza blobs grandes que cobrem muitos
focos (centróide fica longe de focos individuais).

``cell``: a unidade é a **célula** prevista. Métrica padrão em validação de
produtos AF (FIRMS / MODIS): um foco é TP se existir QUALQUER célula prevista
a ≤ R km; uma célula é TP se existir QUALQUER foco a ≤ R km. Mais coerente
para focos densamente agrupados.
"""
"""Aproximação esférica média; precisão ~0,1 % no bounding box do Ceará."""


def _km_per_deg_lon(lat_deg: float) -> float:
    """Conversão lon→km dependente da latitude (graus → km)."""
    return KM_PER_DEG_LAT * float(np.cos(np.deg2rad(lat_deg)))


def _build_grid_centers(bbox: dict, grid_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Centros (lat, lon) de cada célula da grade (matrizes H×W)."""
    h, w = grid_hw
    lat_edges = np.linspace(bbox["min_lat"], bbox["max_lat"], h + 1)
    lon_edges = np.linspace(bbox["min_lon"], bbox["max_lon"], w + 1)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_grid, lon_grid = np.meshgrid(lat_c, lon_c, indexing="ij")
    return lat_grid, lon_grid


def _components_centroids(
    pred: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    *,
    connectivity: int = 2,
) -> Tuple[np.ndarray, List[int]]:
    """
    Devolve ``centroids`` (N, 2) com colunas (lat, lon) e ``sizes``
    (nº de células por componente). ``connectivity=2`` = 8-vizinhos.
    """
    if not pred.any():
        return np.empty((0, 2), dtype=np.float64), []
    structure = np.ones((3, 3), dtype=bool) if connectivity == 2 else None
    lab, n = label(pred, structure=structure)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64), []
    centroids = np.empty((n, 2), dtype=np.float64)
    sizes: List[int] = []
    for k in range(1, n + 1):
        m = lab == k
        sizes.append(int(m.sum()))
        centroids[k - 1, 0] = float(lat_grid[m].mean())
        centroids[k - 1, 1] = float(lon_grid[m].mean())
    return centroids, sizes


def _filter_focos(
    df: pd.DataFrame,
    bbox: dict,
    *,
    day_utc: Optional[Tuple[datetime, datetime]] = None,
) -> pd.DataFrame:
    """Restringe focos ao bbox e (opcional) janela temporal ``[d0, d1)``."""
    if df.empty:
        return df
    mask = (
        (df["lat"] >= bbox["min_lat"])
        & (df["lat"] <= bbox["max_lat"])
        & (df["lon"] >= bbox["min_lon"])
        & (df["lon"] <= bbox["max_lon"])
    )
    if day_utc is not None:
        d0, d1 = day_utc
        mask &= (df["datetime"] >= d0) & (df["datetime"] < d1)
    return df.loc[mask]


def _haversine_like_km(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    lat_b: np.ndarray,
    lon_b: np.ndarray,
) -> np.ndarray:
    """
    Distâncias (km) entre pontos A (N,) e B (M,) — matriz (N, M).
    Equirectangular local: precisão muito boa em ~5° × 5° (Ceará).
    """
    if lat_a.size == 0 or lat_b.size == 0:
        return np.empty((lat_a.size, lat_b.size), dtype=np.float64)
    lat_ref = 0.5 * (float(lat_a.mean()) + float(lat_b.mean()))
    kx = _km_per_deg_lon(lat_ref)
    ky = KM_PER_DEG_LAT
    dlat = (lat_a[:, None] - lat_b[None, :]) * ky
    dlon = (lon_a[:, None] - lon_b[None, :]) * kx
    return np.hypot(dlat, dlon)


@dataclass
class EventCentricMetrics:
    radius_km: float
    n_focos: int
    n_components: int
    tp_for_recall: int
    tp_for_precision: int
    fn_focos: int
    fp_components: int
    precision: float
    recall: float
    f1: float
    component_sizes_median: float
    component_sizes_max: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "ec_radius_km": float(self.radius_km),
            "ec_n_focos": int(self.n_focos),
            "ec_n_components": int(self.n_components),
            "ec_tp_recall": int(self.tp_for_recall),
            "ec_tp_precision": int(self.tp_for_precision),
            "ec_fn_focos": int(self.fn_focos),
            "ec_fp_components": int(self.fp_components),
            "ec_precision": float(self.precision),
            "ec_recall": float(self.recall),
            "ec_f1": float(self.f1),
            "ec_component_size_median": float(self.component_sizes_median),
            "ec_component_size_max": int(self.component_sizes_max),
        }


def evaluate_event_centric(
    pred: np.ndarray,
    df_focos: pd.DataFrame,
    bbox: dict,
    grid_hw: Tuple[int, int],
    *,
    radius_km: float = 3.0,
    day_utc: Optional[Tuple[datetime, datetime]] = None,
    connectivity: int = 2,
    valid_bins: Optional[np.ndarray] = None,
    matching: MatchingMode = "cell",
) -> EventCentricMetrics:
    """
    Calcula métricas event-centric (precision, recall, F1).

    ``pred`` é máscara binária (H, W) já no recorte do ``bbox``.
    ``df_focos`` tem colunas ``lat``, ``lon`` e (opcional) ``datetime``.
    Se ``valid_bins`` for fornecido, ``pred`` é restringido a ``pred & valid_bins``.

    ``matching`` controla a unidade da previsão (ver docstring do módulo).
    """
    h, w = grid_hw
    if pred.shape != (h, w):
        raise ValueError(f"pred shape {pred.shape} != grid {grid_hw}")
    if valid_bins is not None:
        pred = pred & valid_bins

    lat_grid, lon_grid = _build_grid_centers(bbox, grid_hw)
    if matching == "centroid":
        units_latlon, sizes = _components_centroids(
            pred,
            lat_grid,
            lon_grid,
            connectivity=connectivity,
        )
    elif matching == "cell":
        ii = np.flatnonzero(pred.ravel())
        if ii.size > 0:
            units_latlon = np.stack(
                [lat_grid.ravel()[ii], lon_grid.ravel()[ii]],
                axis=1,
            )
            sizes = [1] * units_latlon.shape[0]
        else:
            units_latlon = np.empty((0, 2), dtype=np.float64)
            sizes = []
    else:
        raise ValueError(f"matching inválido: {matching!r}")

    df_in = _filter_focos(df_focos, bbox, day_utc=day_utc)
    n_focos = int(len(df_in))
    n_comp = int(units_latlon.shape[0])

    if n_focos == 0 and n_comp == 0:
        return EventCentricMetrics(
            radius_km=radius_km,
            n_focos=0,
            n_components=0,
            tp_for_recall=0,
            tp_for_precision=0,
            fn_focos=0,
            fp_components=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            component_sizes_median=0.0,
            component_sizes_max=0,
        )

    if n_focos == 0:
        return EventCentricMetrics(
            radius_km=radius_km,
            n_focos=0,
            n_components=n_comp,
            tp_for_recall=0,
            tp_for_precision=0,
            fn_focos=0,
            fp_components=n_comp,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            component_sizes_median=float(np.median(sizes)) if sizes else 0.0,
            component_sizes_max=int(max(sizes)) if sizes else 0,
        )

    if n_comp == 0:
        return EventCentricMetrics(
            radius_km=radius_km,
            n_focos=n_focos,
            n_components=0,
            tp_for_recall=0,
            tp_for_precision=0,
            fn_focos=n_focos,
            fp_components=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            component_sizes_median=0.0,
            component_sizes_max=0,
        )

    lat_f = df_in["lat"].to_numpy(dtype=np.float64)
    lon_f = df_in["lon"].to_numpy(dtype=np.float64)
    lat_c = units_latlon[:, 0]
    lon_c = units_latlon[:, 1]
    D = _haversine_like_km(lat_f, lon_f, lat_c, lon_c)
    within = D <= float(radius_km)

    tp_recall = int(np.sum(np.any(within, axis=1)))
    tp_prec = int(np.sum(np.any(within, axis=0)))
    fn_focos = n_focos - tp_recall
    fp_components = n_comp - tp_prec

    precision = tp_prec / n_comp if n_comp > 0 else 0.0
    recall = tp_recall / n_focos if n_focos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EventCentricMetrics(
        radius_km=float(radius_km),
        n_focos=n_focos,
        n_components=n_comp,
        tp_for_recall=tp_recall,
        tp_for_precision=tp_prec,
        fn_focos=fn_focos,
        fp_components=fp_components,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        component_sizes_median=float(np.median(sizes)) if sizes else 0.0,
        component_sizes_max=int(max(sizes)) if sizes else 0,
    )


def evaluate_event_centric_multi_radius(
    pred: np.ndarray,
    df_focos: pd.DataFrame,
    bbox: dict,
    grid_hw: Tuple[int, int],
    *,
    radii_km: Sequence[float] = (1.5, 3.0, 5.0, 8.0),
    day_utc: Optional[Tuple[datetime, datetime]] = None,
    connectivity: int = 2,
    valid_bins: Optional[np.ndarray] = None,
    matching: MatchingMode = "cell",
) -> Dict[str, Dict[str, float]]:
    """
    Curva F1(R) — repete a avaliação com vários raios para revelar
    a sensibilidade à tolerância espacial. Útil para escolher ``R`` honesto.
    """
    out: Dict[str, Dict[str, float]] = {}
    for r in radii_km:
        m = evaluate_event_centric(
            pred,
            df_focos,
            bbox,
            grid_hw,
            radius_km=float(r),
            day_utc=day_utc,
            connectivity=connectivity,
            valid_bins=valid_bins,
            matching=matching,
        )
        out[f"R={float(r):.1f}km"] = m.to_dict()
    return out


def day_window_utc(day_iso: str) -> Tuple[datetime, datetime]:
    """Janela ``[00:00, 24:00)`` em UTC para uma data ``YYYY-MM-DD``."""
    d0 = datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return d0, d0 + pd.Timedelta(days=1).to_pytimedelta()
