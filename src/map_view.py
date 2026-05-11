"""
Mapa interativo HTML (Folium) — **real vs previsto** por data.

Renderiza num único ``.html`` o bounding box do Ceará, os focos INPE
(círculos vermelhos, popup com hora) e as previsões DTEC (círculos
azuis com `popup` do score). TP / FP / FN são marcados por cor.

Sem dependência de servidor: o ficheiro abre direto no browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd

from src.event_centric import (
    KM_PER_DEG_LAT,
    _build_grid_centers,
    _filter_focos,
    _haversine_like_km,
    _km_per_deg_lon,
)


COLOR_TP = "#238b45"     # verde escuro — acerto
COLOR_FP = "#fd8d3c"     # laranja      — alarme falso
COLOR_FN = "#a50f15"     # vermelho     — fogo não detetado


@dataclass
class MapLayerStats:
    n_focos: int
    n_pred: int
    tp_recall: int
    tp_precision: int
    fn: int
    fp: int


def _classify_focos(
    df: pd.DataFrame,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    radius_km: float,
) -> Tuple[np.ndarray, MapLayerStats, np.ndarray]:
    """Devolve máscara `is_tp` por foco e estatísticas; também `is_tp_pred` (por previsão)."""
    if df.empty or pred_lat.size == 0:
        is_tp_foco = np.zeros(len(df), dtype=bool)
        is_tp_pred = np.zeros(pred_lat.size, dtype=bool)
        return is_tp_foco, MapLayerStats(
            n_focos=int(len(df)),
            n_pred=int(pred_lat.size),
            tp_recall=0,
            tp_precision=0,
            fn=int(len(df)),
            fp=int(pred_lat.size),
        ), is_tp_pred
    D = _haversine_like_km(
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        pred_lat,
        pred_lon,
    )
    is_tp_foco = np.any(D <= float(radius_km), axis=1)
    is_tp_pred = np.any(D <= float(radius_km), axis=0)
    return (
        is_tp_foco,
        MapLayerStats(
            n_focos=int(len(df)),
            n_pred=int(pred_lat.size),
            tp_recall=int(is_tp_foco.sum()),
            tp_precision=int(is_tp_pred.sum()),
            fn=int((~is_tp_foco).sum()),
            fp=int((~is_tp_pred).sum()),
        ),
        is_tp_pred,
    )


def _pred_latlon_from_mask(
    pred_mask: np.ndarray,
    bbox: dict,
    grid_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    lat_g, lon_g = _build_grid_centers(bbox, grid_hw)
    ii = np.flatnonzero(pred_mask.ravel())
    return lat_g.ravel()[ii], lon_g.ravel()[ii]


def build_map(
    df_focos: pd.DataFrame,
    pred_mask: np.ndarray,
    bbox: dict,
    grid_hw: Tuple[int, int],
    *,
    day_iso: str,
    radius_km: float = 10.0,
    valid_bins: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> folium.Map:
    """Constrói o objeto ``folium.Map`` para uma data."""
    if valid_bins is not None:
        pred_mask = pred_mask & valid_bins

    center_lat = 0.5 * (bbox["min_lat"] + bbox["max_lat"])
    center_lon = 0.5 * (bbox["min_lon"] + bbox["max_lon"])
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Bounding box do Ceará
    folium.Rectangle(
        bounds=[(bbox["min_lat"], bbox["min_lon"]), (bbox["max_lat"], bbox["max_lon"])],
        color="#444",
        weight=1,
        fill=False,
        opacity=0.6,
        popup="BBOX Ceará (avaliação DTEC)",
    ).add_to(m)

    d0 = datetime.strptime(day_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = d0 + pd.Timedelta(days=1)
    df_day = _filter_focos(df_focos, bbox, day_utc=(d0, d1))
    pred_lat, pred_lon = _pred_latlon_from_mask(pred_mask, bbox, grid_hw)
    is_tp_foco, stats, is_tp_pred = _classify_focos(df_day, pred_lat, pred_lon, radius_km)

    fg_focos_tp = folium.FeatureGroup(name=f"INPE foco (TP): {int(stats.tp_recall)}", show=True)
    fg_focos_fn = folium.FeatureGroup(name=f"INPE foco (FN): {int(stats.fn)}", show=True)
    for i, row in df_day.reset_index(drop=True).iterrows():
        tp = bool(is_tp_foco[i]) if i < len(is_tp_foco) else False
        color = COLOR_TP if tp else COLOR_FN
        group = fg_focos_tp if tp else fg_focos_fn
        ts = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d %H:%M UTC")
        popup = f"<b>Foco INPE</b><br/>{ts}<br/>lat={row['lat']:.4f}, lon={row['lon']:.4f}<br/>{'TP' if tp else 'FN'}"
        folium.CircleMarker(
            location=(float(row["lat"]), float(row["lon"])),
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(group)
    fg_focos_tp.add_to(m)
    fg_focos_fn.add_to(m)

    fg_pred_tp = folium.FeatureGroup(name=f"DTEC previsão (TP): {int(stats.tp_precision)}", show=True)
    fg_pred_fp = folium.FeatureGroup(name=f"DTEC previsão (FP): {int(stats.fp)}", show=True)
    for i in range(pred_lat.size):
        tp = bool(is_tp_pred[i]) if i < is_tp_pred.size else False
        color = COLOR_TP if tp else COLOR_FP
        group = fg_pred_tp if tp else fg_pred_fp
        popup = (
            f"<b>DTEC previsão</b><br/>"
            f"lat={pred_lat[i]:.4f}, lon={pred_lon[i]:.4f}<br/>"
            f"{'TP' if tp else 'FP'} (R={radius_km:.1f} km)"
        )
        folium.CircleMarker(
            location=(float(pred_lat[i]), float(pred_lon[i])),
            radius=5,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            popup=folium.Popup(popup, max_width=240),
        ).add_to(group)
    fg_pred_tp.add_to(m)
    fg_pred_fp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legenda + métricas
    precision = stats.tp_precision / max(stats.n_pred, 1)
    recall = stats.tp_recall / max(stats.n_focos, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    legend_html = f"""
<div style="position: fixed; bottom: 20px; left: 20px; z-index:9999;
            background: white; padding: 10px 12px; border:2px solid #444;
            font: 12px/1.35 -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
  <b>{title or 'DTEC vs INPE'} — {day_iso}</b><br/>
  <span style="color:{COLOR_TP}">●</span> TP (acerto)
  &nbsp;<span style="color:{COLOR_FP}">●</span> FP (alarme falso)
  &nbsp;<span style="color:{COLOR_FN}">●</span> FN (foco não detectado)<br/>
  <hr style="margin:6px 0; border:0; border-top:1px solid #ccc"/>
  Focos INPE: <b>{stats.n_focos}</b> &nbsp; Previstos: <b>{stats.n_pred}</b><br/>
  TP (recall): <b>{stats.tp_recall}</b> &nbsp; FN: <b>{stats.fn}</b><br/>
  TP (precisão): <b>{stats.tp_precision}</b> &nbsp; FP: <b>{stats.fp}</b><br/>
  R={radius_km:.1f} km &nbsp; P=<b>{precision:.3f}</b> R=<b>{recall:.3f}</b> F1=<b>{f1:.3f}</b>
</div>
"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def save_map(
    out_path: Path,
    df_focos: pd.DataFrame,
    pred_mask: np.ndarray,
    bbox: dict,
    grid_hw: Tuple[int, int],
    *,
    day_iso: str,
    radius_km: float = 10.0,
    valid_bins: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Path:
    """Constrói e grava o mapa em ``out_path`` (HTML)."""
    m = build_map(
        df_focos, pred_mask, bbox, grid_hw,
        day_iso=day_iso, radius_km=radius_km, valid_bins=valid_bins, title=title,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    return out_path.resolve()


def build_multi_date_index(
    map_paths: Sequence[Tuple[str, Path]],
    out_path: Path,
    *,
    title: str = "DTEC — mapas previstos vs reais",
) -> Path:
    """Gera um índice HTML simples com links para cada data."""
    items = "\n".join(
        f'<li><a href="{p.name}">{day}</a></li>'
        for day, p in map_paths
    )
    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, Arial, sans-serif;
       max-width: 720px; margin: 2em auto; padding: 0 1em; color:#222; }}
h1 {{ font-size: 1.6em; margin-bottom: 0.2em; }}
.lead {{ color:#555; margin-bottom: 1em; }}
ul {{ padding-left: 1.3em; }}
li {{ margin: 0.25em 0; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="lead">Selecione uma data para abrir o mapa interativo (Folium).
Os pontos representam focos INPE (real) e previsões DTEC; cores TP/FP/FN
seguem a convenção do mapa.</p>
<ul>
{items}
</ul>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path.resolve()
