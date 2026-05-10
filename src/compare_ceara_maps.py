"""
Mapas do Ceará (bbox IBGE) + focos reais vs previstos, e exportação de métricas agregadas.

Usado por `compare_goes_unsupervised_days` e `compare_st_hypernet_days`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config.ceara_config import CEARA_BBOX


def display_positive_mask(
    pred_g: np.ndarray,
    base_thr: float,
    max_positive_fraction: float = 0.2,
) -> Tuple[np.ndarray, float, str]:
    """
    Máscara binária **só para plotagem** (painel verde / pontos no mapa). Se muitas
    células ficam ≥ ``base_thr``, sobe o limiar; se ainda saturado (ex.: valores
    idênticos), mantém apenas as ``max_positive_fraction × N`` células de maior score.

    Métricas IoU/P/R no caller devem continuar usando ``pred_threshold`` sem esta máscara.
    """
    g = np.asarray(pred_g, dtype=np.float64)
    flat = g.ravel()
    n = flat.size
    if n == 0:
        return np.zeros_like(g, dtype=np.float32), float(base_thr), "empty"
    base_thr = float(base_thr)
    if float((flat >= base_thr).mean()) <= max_positive_fraction:
        return (g >= base_thr).astype(np.float32), base_thr, "base"
    t = float(np.quantile(flat, 1.0 - max_positive_fraction))
    t = max(t, base_thr)
    if float((flat >= t).mean()) <= max_positive_fraction:
        return (g >= t).astype(np.float32), t, "adapted_quantile"
    k = max(1, int(math.floor(max_positive_fraction * n)))
    idx = np.argpartition(flat, n - k)[n - k :]
    m = np.zeros(n, dtype=np.float32)
    m[idx] = 1.0
    thr_used = float(flat[idx].min())
    return m.reshape(g.shape), max(thr_used, base_thr), "adapted_topk"


def pred_points_from_display_mask(
    pred_g: np.ndarray,
    n_lat: int,
    n_lon: int,
    mask: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Centros (lat, lon, score) das células ativas na máscara de visualização."""
    pts: List[Tuple[float, float, float]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            if float(mask[i, j]) > 0.5:
                sc = float(pred_g[i, j])
                la, lo = cell_center_lat_lon(i, j, n_lat, n_lon)
                pts.append((la, lo, sc))
    pts.sort(key=lambda x: -x[2])
    return pts


def _day_key(day) -> Any:
    return pd.Timestamp(day).normalize().date()


def cell_center_lat_lon(i: int, j: int, n_lat: int, n_lon: int) -> Tuple[float, float]:
    lat = CEARA_BBOX["min_lat"] + (float(i) + 0.5) * (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / max(n_lat, 1)
    lon = CEARA_BBOX["min_lon"] + (float(j) + 0.5) * (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / max(n_lon, 1)
    return float(lat), float(lon)


def real_foci_points_day(dfx: pd.DataFrame, day) -> List[Tuple[float, float]]:
    """Todos os focos (lat, lon) daquele dia civil."""
    dkey = _day_key(day)
    mask = pd.to_datetime(dfx["date"]).dt.normalize().dt.date == dkey
    sub = dfx.loc[mask]
    out: List[Tuple[float, float]] = []
    for _, row in sub.iterrows():
        out.append((float(row["lat"]), float(row["lon"])))
    return out


def pred_grid_points(pred_g: np.ndarray, n_lat: int, n_lon: int, pred_threshold: float) -> List[Tuple[float, float, float]]:
    """Centros de células com score >= limiar (lat, lon, score)."""
    pts: List[Tuple[float, float, float]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            sc = float(pred_g[i, j])
            if sc >= pred_threshold:
                la, lo = cell_center_lat_lon(i, j, n_lat, n_lon)
                pts.append((la, lo, sc))
    return pts


def save_ceara_folium_map(
    out_html: Path,
    title: str,
    real_pts: Sequence[Tuple[float, float]],
    pred_pts: Sequence[Tuple[float, float, float]],
    pred_threshold: float,
    display_threshold: float | None = None,
) -> None:
    import folium
    from folium.plugins import Fullscreen

    from src.satellite import satellite_layer_for_folium

    mid_lat = (CEARA_BBOX["min_lat"] + CEARA_BBOX["max_lat"]) / 2.0
    mid_lon = (CEARA_BBOX["min_lon"] + CEARA_BBOX["max_lon"]) / 2.0
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=7, control_scale=True, tiles=None)
    url, attr, opts = satellite_layer_for_folium("esri_satellite")
    folium.TileLayer(
        url,
        attr=attr,
        name="Imagem de satélite (ESRI World Imagery)",
        overlay=False,
        control=True,
        **{k: v for k, v in opts.items() if k != "name"},
    ).add_to(m)

    fg_real = folium.FeatureGroup(name="Queimadas reais (focos)", show=True)
    for lat, lon in real_pts:
        fg_real.add_child(
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="crimson",
                weight=1,
                fill=True,
                fill_color="crimson",
                fill_opacity=0.75,
            )
        )

    thr_lab = float(display_threshold) if display_threshold is not None else float(pred_threshold)
    fg_pred = folium.FeatureGroup(name=f"Previsto (células vis. ≥ {thr_lab:.2f})", show=True)
    for lat, lon, sc in pred_pts:
        r = 4.0 + min(14.0, sc * 12.0)
        fg_pred.add_child(
            folium.CircleMarker(
                location=[lat, lon],
                radius=r,
                color="darkgreen",
                weight=1,
                fill=True,
                fill_color="limegreen",
                fill_opacity=0.45,
            )
        )

    m.add_child(fg_real)
    m.add_child(fg_pred)

    folium.Rectangle(
        bounds=[
            [CEARA_BBOX["min_lat"], CEARA_BBOX["min_lon"]],
            [CEARA_BBOX["max_lat"], CEARA_BBOX["max_lon"]],
        ],
        color="#003366",
        weight=2,
        fill=False,
        popup="Limite da grade (Ceará bbox)",
    ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def map_points_to_json_fields(
    real_pts: Sequence[Tuple[float, float]],
    pred_pts: Sequence[Tuple[float, float, float]],
) -> Dict[str, Any]:
    """Listas JSON-seriáveis para reconstruir mapas (ex.: dashboard) sem reler o cubo."""
    return {
        "map_real_latlon": [[float(a), float(b)] for a, b in real_pts],
        "map_pred_latlon_score": [[float(lat), float(lon), float(sc)] for lat, lon, sc in pred_pts],
    }


def build_ceara_folium_real_pred_goes(
    *,
    real_latlon: Sequence[Sequence[float]],
    pred_latlon_score: Sequence[Sequence[float]],
    goes_latlon: Sequence[Tuple[float, float]] | None = None,
    pred_layer_label: str = "Previsto (modelo — células)",
    basemap_id: str = "esri_satellite",
    basemap_override: Tuple[str, str, dict] | None = None,
) -> Any:
    """
    Mapa Folium: fundo de imagem de satélite + focos reais (BD) + previsto na grade
    + opcionalmente detecções GOES-16/-19 (INPE) no mesmo dia.
    """
    import folium
    from folium.plugins import Fullscreen

    from src.satellite import satellite_layer_for_folium

    mid_lat = (CEARA_BBOX["min_lat"] + CEARA_BBOX["max_lat"]) / 2.0
    mid_lon = (CEARA_BBOX["min_lon"] + CEARA_BBOX["max_lon"]) / 2.0
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=7, control_scale=True, tiles=None)

    if basemap_override is not None:
        url, attr, opts = basemap_override
    else:
        url, attr, opts = satellite_layer_for_folium(basemap_id)
    folium.TileLayer(
        url,
        attr=attr,
        name="Imagem de satélite (base)",
        overlay=False,
        control=True,
        **{k: v for k, v in opts.items() if k != "name"},
    ).add_to(m)

    fg_real = folium.FeatureGroup(name="Focos reais (referência do dia)", show=True)
    for pair in real_latlon:
        if len(pair) < 2:
            continue
        lat, lon = float(pair[0]), float(pair[1])
        fg_real.add_child(
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="crimson",
                weight=1,
                fill=True,
                fill_color="crimson",
                fill_opacity=0.82,
                popup=folium.Popup("Foco real (BDQueimadas / cubo)", max_width=200),
            )
        )

    fg_pred = folium.FeatureGroup(name=pred_layer_label, show=True)
    for triple in pred_latlon_score:
        if len(triple) < 2:
            continue
        lat, lon = float(triple[0]), float(triple[1])
        sc = float(triple[2]) if len(triple) > 2 else 0.5
        r = 4.0 + min(14.0, sc * 12.0)
        fg_pred.add_child(
            folium.CircleMarker(
                location=[lat, lon],
                radius=r,
                color="darkgreen",
                weight=1,
                fill=True,
                fill_color="limegreen",
                fill_opacity=0.5,
                popup=folium.Popup(f"Previsto (score≈{sc:.3f})", max_width=200),
            )
        )

    m.add_child(fg_real)
    m.add_child(fg_pred)

    if goes_latlon:
        fg_g = folium.FeatureGroup(name="Detecções térmicas GOES-16/-19 (INPE)", show=True)
        for lat, lon in goes_latlon:
            fg_g.add_child(
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=3,
                    color="darkorange",
                    weight=1,
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.75,
                    popup=folium.Popup("GOES (INPE)", max_width=160),
                )
            )
        m.add_child(fg_g)

    folium.Rectangle(
        bounds=[
            [CEARA_BBOX["min_lat"], CEARA_BBOX["min_lon"]],
            [CEARA_BBOX["max_lat"], CEARA_BBOX["max_lon"]],
        ],
        color="#003366",
        weight=2,
        fill=False,
        popup="Limite da grade (Ceará bbox)",
    ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_ceara_map_png(
    out_png: Path,
    title: str,
    real_pts: Sequence[Tuple[float, float]],
    pred_pts: Sequence[Tuple[float, float, float]],
    metrics: Dict[str, Any],
    pred_raster: np.ndarray | None = None,
    display_thr: float | None = None,
    satellite_basemap: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    mid_lat = (CEARA_BBOX["min_lat"] + CEARA_BBOX["max_lat"]) / 2.0
    aspect = 1.0 / max(0.2, math.cos(math.radians(mid_lat)))

    fig, ax = plt.subplots(figsize=(11, 11 * aspect))
    extent = (
        CEARA_BBOX["min_lon"],
        CEARA_BBOX["max_lon"],
        CEARA_BBOX["min_lat"],
        CEARA_BBOX["max_lat"],
    )
    ax.set_xlim(CEARA_BBOX["min_lon"], CEARA_BBOX["max_lon"])
    ax.set_ylim(CEARA_BBOX["min_lat"], CEARA_BBOX["max_lat"])
    ax.set_aspect("equal")

    if satellite_basemap:
        try:
            import contextily as cx

            cx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=cx.providers.Esri.WorldImagery,
                attribution_size=6,
            )
        except Exception:
            ax.set_facecolor("#e8eef2")

    if pred_raster is not None and pred_raster.size > 0:
        pr = np.asarray(pred_raster, dtype=np.float32)
        lo = float(np.percentile(pr, 6.0))
        hi = float(np.percentile(pr, 97.5))
        if hi <= lo + 1e-9:
            hi = float(pr.max()) + 1e-6
        alpha_r = 0.38 if satellite_basemap else 0.42
        im_bg = ax.imshow(
            pr,
            origin="lower",
            extent=(extent[0], extent[1], extent[2], extent[3]),
            aspect="equal",
            cmap="magma",
            vmin=lo,
            vmax=hi,
            alpha=alpha_r,
            interpolation="nearest",
            zorder=1,
        )
        cbar = plt.colorbar(im_bg, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Score (contraste local)", fontsize=8)

    if real_pts:
        ax.scatter(
            [p[1] for p in real_pts],
            [p[0] for p in real_pts],
            c="crimson",
            s=22,
            alpha=0.75,
            label=f"Real (n={len(real_pts)})",
            zorder=5,
            edgecolors="darkred",
            linewidths=0.2,
        )
    if pred_pts:
        ax.scatter(
            [p[1] for p in pred_pts],
            [p[0] for p in pred_pts],
            c="#2ecc71",
            s=[32 + 55 * min(1.0, p[2]) for p in pred_pts],
            alpha=0.55,
            edgecolors="#145a32",
            linewidths=0.35,
            label=f"Previsto células (n={len(pred_pts)})",
            zorder=4,
        )

    rect = plt.Rectangle(
        (CEARA_BBOX["min_lon"], CEARA_BBOX["min_lat"]),
        CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"],
        CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"],
        fill=False,
        edgecolor="yellow",
        linewidth=2.2,
        zorder=6,
    )
    ax.add_patch(rect)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.12 if satellite_basemap else 0.28, linestyle="--", color="white" if satellite_basemap else "gray")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    dthr = display_thr if display_thr is not None else metrics.get("pred_threshold")
    mode = metrics.get("pred_display_mode", "")
    thr_note = f"plot: previsto vis. ≥ {float(dthr):.2f}" if dthr is not None else ""
    if mode and mode != "base":
        thr_note = f"{thr_note}  [{mode}]" if thr_note else f"[{mode}]"
    st = (
        f"IoU={metrics.get('iou', 0):.3f}  P={metrics.get('precision', 0):.3f}  R={metrics.get('recall', 0):.3f}  "
        f"TP={metrics.get('tp', 0)} FP={metrics.get('fp', 0)} FN={metrics.get('fn', 0)}"
    )
    ax.set_xlim(CEARA_BBOX["min_lon"], CEARA_BBOX["max_lon"])
    ax.set_ylim(CEARA_BBOX["min_lat"], CEARA_BBOX["max_lat"])
    ax.set_title(f"{title}\n{thr_note}\n{st}", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def build_aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    ious = [float(r["iou"]) for r in rows]
    precs = [float(r["precision"]) for r in rows]
    recs = [float(r["recall"]) for r in rows]
    tps = sum(int(r.get("tp", 0)) for r in rows)
    fps = sum(int(r.get("fp", 0)) for r in rows)
    fns = sum(int(r.get("fn", 0)) for r in rows)
    return {
        "n_days": len(rows),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
        "total_tp": int(tps),
        "total_fp": int(fps),
        "total_fn": int(fns),
        "micro_precision": float(tps / max(1, tps + fps)),
        "micro_recall": float(tps / max(1, tps + fns)),
    }


def write_aggregate_metrics_json(rows: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    agg = build_aggregate_metrics(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg


def save_all_metrics_outputs(out: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """CSV por dia, agregado JSON e `metrics_by_day.json` completo com bloco agregado."""
    summary = dict(summary)
    summary["metrics_aggregate"] = build_aggregate_metrics(rows)
    write_metrics_csv(rows, out / "metrics_by_day.csv")
    write_aggregate_metrics_json(rows, out / "metrics_aggregate.json")
    with open(out / "metrics_by_day.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
