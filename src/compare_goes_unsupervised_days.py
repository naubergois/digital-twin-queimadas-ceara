"""
Compara dias com incêndios reais (focos na grade) vs predição GOES não supervisionada
(Isolation Forest / fallback em cubo proxy ABI) e salva figuras PNG + JSON de métricas.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config.ceara_config import CEARA_BBOX

from src.goes_unsupervised_twin import (
    GOESUnsupervisedConfig,
    predict_unsupervised_anomaly_cube,
    _build_stack_from_xarray,
    _load_proxy_dataset_from_netcdf,
)
from src.pyro_caatinga import ClimatologyResidualFrontEnd, PyroCaatingaConfig, build_goes_proxy_cube


def _prepare_foci_df(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    for col in ["datetime", "data_hora", "data_hora_gmt", "acq_date"]:
        if col in dfx.columns:
            dfx["datetime"] = pd.to_datetime(dfx[col], errors="coerce")
            break
    if "datetime" not in dfx.columns:
        dfx["datetime"] = pd.NaT
    dfx = dfx.dropna(subset=["datetime"])
    dfx["date"] = dfx["datetime"].dt.normalize().dt.date
    if "lat" not in dfx.columns or "lon" not in dfx.columns:
        raise ValueError("DataFrame precisa de colunas lat/lon (ou latitude/longitude).")
    dfx["lat"] = pd.to_numeric(dfx["lat"], errors="coerce")
    dfx["lon"] = pd.to_numeric(dfx["lon"], errors="coerce")
    dfx = dfx.dropna(subset=["lat", "lon"])
    m = (
        (dfx["lat"] >= CEARA_BBOX["min_lat"])
        & (dfx["lat"] <= CEARA_BBOX["max_lat"])
        & (dfx["lon"] >= CEARA_BBOX["min_lon"])
        & (dfx["lon"] <= CEARA_BBOX["max_lon"])
    )
    return dfx.loc[m].copy()


def _lat_lon_to_grid(lat: float, lon: float, n_lat: int, n_lon: int) -> Tuple[int, int]:
    i = int((n_lat - 1) * (lat - CEARA_BBOX["min_lat"]) / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]))
    j = int((n_lon - 1) * (lon - CEARA_BBOX["min_lon"]) / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]))
    return max(0, min(i, n_lat - 1)), max(0, min(j, n_lon - 1))


def _calendar_date(day) -> Any:
    """Normaliza numpy.datetime64 / str / date para comparação estável."""
    return pd.Timestamp(day).normalize().date()


def fire_days_in_cube_span(
    counts: pd.Series,
    times: pd.DatetimeIndex,
    max_days: int,
) -> List[Any]:
    """
    Dias com focos que caem dentro do eixo temporal do cubo (evita predição vazia
    quando max_days_history corta o ano e os 'top dias' ficam fora da janela).

    Se ``max_days <= 0``, retorna **todos** esses dias, ordenados do mais antigo ao
    mais recente (bom para avaliar o ano inteiro). Caso contrário, ordena por volume
    de focos (maior primeiro) e corta em ``max_days``.
    """
    if len(times) == 0:
        return []
    span_lo = pd.Timestamp(times[0]).normalize().date()
    span_hi = pd.Timestamp(times[-1]).normalize().date()
    ranked = [d for d in counts.index if int(counts[d]) > 0 and span_lo <= _calendar_date(d) <= span_hi]
    if int(max_days) <= 0:
        ranked.sort(key=lambda d: _calendar_date(d))
        return ranked
    ranked.sort(key=lambda d: int(counts[d]), reverse=True)
    return ranked[: max(1, int(max_days))]


def normalize_pred_grid_for_display(pred_g: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Escala a grade prevista para [0, 1] de forma mais estável que dividir só pelo máximo:
    um único pixel muito alto não 'apaga' o resto do dia (comum em dias com pouco sinal).
    """
    g = np.asarray(pred_g, dtype=np.float32)
    flat = g.ravel()
    mx = float(np.max(flat)) if flat.size else 0.0
    if mx <= 1e-12:
        return g, {"pred_raw_max": mx, "norm_denom": 0.0, "norm": "all_zero"}
    p995 = float(np.percentile(flat, 99.5))
    denom = max(p995, mx * 0.12, 1e-9)
    out = np.clip(g / denom, 0.0, 1.0).astype(np.float32)
    return out, {"pred_raw_max": mx, "norm_denom": float(denom), "p995": float(p995), "norm": "p995_floor"}


def daily_real_grid(df: pd.DataFrame, day, n_lat: int, n_lon: int) -> np.ndarray:
    g = np.zeros((n_lat, n_lon), dtype=np.float32)
    dkey = _calendar_date(day)
    mask = pd.to_datetime(df["date"]).dt.normalize().dt.date == dkey
    sub = df.loc[mask]
    for _, row in sub.iterrows():
        i, j = _lat_lon_to_grid(float(row["lat"]), float(row["lon"]), n_lat, n_lon)
        g[i, j] = 1.0
    return g


def daily_pred_grid(
    anomaly_cube: np.ndarray,
    times: pd.DatetimeIndex,
    day,
) -> Tuple[np.ndarray, int]:
    """Máximo temporal dos scores de anomalia nos instantes daquele dia civil."""
    dkey = _calendar_date(day)
    idx = [t for t in range(len(times)) if pd.Timestamp(times[t]).normalize().date() == dkey]
    if not idx:
        return np.zeros((anomaly_cube.shape[1], anomaly_cube.shape[2]), dtype=np.float32), 0
    return np.max(anomaly_cube[idx], axis=0).astype(np.float32), len(idx)


def binary_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, float]:
    yt = (y_true > 0).astype(np.uint8).ravel()
    yp = (y_pred_binary > 0).astype(np.uint8).ravel()
    tp = int(np.logical_and(yt == 1, yp == 1).sum())
    fp = int(np.logical_and(yt == 0, yp == 1).sum())
    fn = int(np.logical_and(yt == 1, yp == 0).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    return {"iou": float(iou), "precision": float(prec), "recall": float(rec), "tp": tp, "fp": fp, "fn": fn}


def adaptive_threshold_day_grid(
    real_g: np.ndarray,
    pred_g: np.ndarray,
    base_thr: float,
    min_recall: float = 0.12,
    min_precision_floor: float = 0.06,
    fp_penalty: float = 0.42,
) -> Tuple[float, Dict[str, Any]]:
    """
    Limiar **adaptativo por dia** sobre a grade normalizada: maximiza combinação
    precisão / IoU com penalidade explícita de FP (taxa sobre células negativas),
    com recall mínimo e piso de precisão. O ``base_thr`` do CLI atua como piso
    (nunca se usa um limiar mais baixo que ele, para não aumentar FP vs. o mínimo
    pedido pelo usuário).

    Isto é um pequeno "modelo adaptativo" em regra (otimização em grade de
    limiares); não altera o treino ST/GOES, só a decisão binária por dia.
    """
    real = (np.asarray(real_g) > 0).astype(np.uint8)
    pred = np.asarray(pred_g, dtype=np.float32)
    flat = pred.ravel().astype(np.float64)
    n = int(flat.size)
    n_pos = int(real.sum())
    n_neg = max(1, n - n_pos)
    base_thr = float(base_thr)

    meta: Dict[str, Any] = {
        "adaptive_model": "grid_precision_iou_minus_fp_rate",
        "min_recall": float(min_recall),
        "min_precision_floor": float(min_precision_floor),
        "fp_penalty": float(fp_penalty),
        "base_thr": base_thr,
    }

    if n_pos == 0:
        t_cut = float(np.max(flat)) + 0.02
        t_cut = float(np.clip(max(t_cut, base_thr), base_thr, 1.0))
        pb = (pred >= t_cut).astype(np.float32)
        m = binary_metrics(real_g, pb)
        meta.update({"mode": "no_positive_labels", "chosen": t_cut, **m})
        return t_cut, meta

    lo, hi = float(np.min(flat)), float(np.max(flat))
    cands: set[float] = {base_thr}
    for q in np.linspace(0.30, 0.995, 32):
        cands.add(float(np.quantile(flat, q)))
    if hi > lo + 1e-9:
        for x in np.linspace(lo, hi, 18):
            cands.add(float(x))
    cands.add(hi)
    cands_sorted = sorted(cands)

    def score_tuple(t: float) -> Tuple[float, float, Dict[str, float]]:
        pb = (pred >= float(t)).astype(np.float32)
        m = binary_metrics(real_g, pb)
        fp_rate = m["fp"] / n_neg
        score = 0.48 * m["precision"] + 0.40 * m["iou"] - fp_penalty * fp_rate
        return score, float(t), m

    feasible: List[Tuple[float, float, Dict[str, float]]] = []
    for t in cands_sorted:
        sc, tt, m = score_tuple(t)
        if m["recall"] + 1e-6 < min_recall:
            continue
        if m["precision"] + 1e-6 < min_precision_floor and m["tp"] > 0:
            continue
        feasible.append((sc, tt, m))

    if not feasible:
        for relax in (0.7, 0.45, 0.2, 0.0):
            mr = max(0.0, min_recall * relax)
            mp = max(0.02, min_precision_floor * (0.5 + 0.5 * relax))
            feasible = []
            for t in cands_sorted:
                sc, tt, m = score_tuple(t)
                if m["recall"] + 1e-6 < mr:
                    continue
                if m["precision"] + 1e-6 < mp and m["tp"] > 0:
                    continue
                feasible.append((sc, tt, m))
            if feasible:
                meta["relaxed_min_recall"] = mr
                meta["relaxed_min_precision_floor"] = mp
                break

    if not feasible:
        t0 = base_thr
        _, _, m0 = score_tuple(t0)
        meta.update({"mode": "fallback_base_only", "chosen": t0, **m0})
        return max(t0, base_thr), meta

    best_score, best_t, best_m = max(feasible, key=lambda x: (x[0], x[1]))
    chosen = max(best_t, base_thr)
    if chosen != best_t:
        _, _, best_m = score_tuple(chosen)
    meta.update({"mode": "adaptive_grid", "chosen": float(chosen), "objective_score": float(best_score), **best_m})
    return float(chosen), meta


def _extent_lon_lat() -> Tuple[float, float, float, float]:
    return (
        CEARA_BBOX["min_lon"],
        CEARA_BBOX["max_lon"],
        CEARA_BBOX["min_lat"],
        CEARA_BBOX["max_lat"],
    )


def save_comparison_figure(
    real_g: np.ndarray,
    pred_g: np.ndarray,
    pred_threshold: float,
    day_str: str,
    metrics: Dict[str, Any],
    out_path: Path,
    display_mask: np.ndarray | None = None,
    display_thr: float | None = None,
    display_mode: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    from src.compare_ceara_maps import display_positive_mask

    extent = _extent_lon_lat()
    pg = np.asarray(pred_g, dtype=np.float32)
    if display_mask is None or display_thr is None or display_mode is None:
        display_mask, display_thr, display_mode = display_positive_mask(pg, float(pred_threshold))
    pred_bin_vis = np.asarray(display_mask, dtype=np.float32)
    rgb = np.zeros((*real_g.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(real_g, 0, 1)
    rgb[..., 1] = pred_bin_vis
    rgb[..., 2] = 0.12

    p_lo = float(np.percentile(pg, 5.0))
    p_hi = float(np.percentile(pg, 97.0))
    if p_hi <= p_lo + 1e-8:
        p_lo, p_hi = 0.0, max(float(pg.max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.55), constrained_layout=True)
    im0 = axes[0].imshow(
        real_g, origin="lower", extent=extent, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation="nearest"
    )
    axes[0].set_title("Real (focos na grade)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(
        pg,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="cividis",
        vmin=p_lo,
        vmax=p_hi,
        interpolation="nearest",
    )
    axes[1].set_title("Previsto (score; contraste por percentis)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    axes[2].set_title(f"Sobreposição (G = vis ≥{float(display_thr):.2f}, {display_mode})")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.suptitle(
        f"{day_str}  |  IoU={metrics['iou']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"focos={metrics.get('n_focos', 0)}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_cube_and_times(
    df: pd.DataFrame,
    cfg: GOESUnsupervisedConfig,
    netcdf_path: str | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, pd.DatetimeIndex]:
    if netcdf_path and Path(netcdf_path).is_file():
        ds = _load_proxy_dataset_from_netcdf(netcdf_path)
        bt7, bt14, delta, res_d = _build_stack_from_xarray(ds)
        if res_d is None:
            times = pd.DatetimeIndex(ds["time"].values)
            front_b7 = ClimatologyResidualFrontEnd(ewma_lambda=0.05)
            front_b14 = ClimatologyResidualFrontEnd(ewma_lambda=0.05)
            res_b7, _ = front_b7.fit_transform(np.asarray(ds["bt7"].values, dtype=np.float32), times)
            res_b14, _ = front_b14.fit_transform(np.asarray(ds["bt14"].values, dtype=np.float32), times)
            res_d = (res_b7 - res_b14).astype(np.float32)
        times = pd.DatetimeIndex(ds["time"].values)
    else:
        pyro_cfg = PyroCaatingaConfig(
            grid_resolution=cfg.grid_resolution,
            frame_minutes=cfg.frame_minutes,
            max_days_history=cfg.max_days_history,
        )
        ds = build_goes_proxy_cube(df, pyro_cfg)
        times = pd.DatetimeIndex(ds["time"].values)
        front_b7 = ClimatologyResidualFrontEnd(ewma_lambda=pyro_cfg.ewma_lambda)
        front_b14 = ClimatologyResidualFrontEnd(ewma_lambda=pyro_cfg.ewma_lambda)
        res_b7, _ = front_b7.fit_transform(np.asarray(ds["bt7"].values, dtype=np.float32), times)
        res_b14, _ = front_b14.fit_transform(np.asarray(ds["bt14"].values, dtype=np.float32), times)
        res_d = (res_b7 - res_b14).astype(np.float32)
        bt7 = np.asarray(ds["bt7"].values, dtype=np.float32)
        bt14 = np.asarray(ds["bt14"].values, dtype=np.float32)
        delta = np.asarray(ds["delta_bt"].values, dtype=np.float32)
    return bt7, bt14, delta, res_d, times


def compare_fire_days_and_save_figures(
    df: pd.DataFrame,
    output_dir: str | Path,
    cfg: GOESUnsupervisedConfig | None = None,
    netcdf_path: str | None = None,
    max_days: int = 12,
    pred_threshold: float = 0.35,
    file_prefix: str = "goes_unsup",
    year: int | None = None,
    adaptive_threshold: bool = True,
    adaptive_min_recall: float = 0.12,
    adaptive_min_precision: float = 0.06,
    adaptive_fp_penalty: float = 0.42,
) -> Dict[str, Any]:
    """
    Seleciona dias com focos reais, compara grade real vs predição não supervisionada
    por dia e grava PNGs + `metrics_by_day.json` em ``output_dir``.
    """
    cfg = cfg or GOESUnsupervisedConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dfx = _prepare_foci_df(df)
    if dfx.empty:
        raise ValueError("Sem focos válidos no bounding box do Ceará.")
    if year is not None:
        dfx = dfx[pd.to_datetime(dfx["datetime"]).dt.year == int(year)].copy()
    if dfx.empty:
        raise ValueError(f"Sem focos após filtro year={year}.")

    bt7, bt14, delta, res_d, times = build_cube_and_times(dfx, cfg, netcdf_path)
    anomaly = predict_unsupervised_anomaly_cube(bt7, bt14, delta, res_d, cfg)
    T, n_lat, n_lon = anomaly.shape

    counts = dfx.groupby("date").size().sort_values(ascending=False)
    fire_days = fire_days_in_cube_span(counts, times, max_days)
    if not fire_days:
        raise ValueError(
            "Nenhum dia com focos intersecta o eixo temporal do cubo. "
            "Aumente --max-days-history (GOES) ou max_days_history no config (ST)."
        )

    rows: List[Dict[str, Any]] = []
    for day in fire_days:
        real_g = daily_real_grid(dfx, day, n_lat, n_lon)
        pred_raw, n_frames = daily_pred_grid(anomaly, times, day)
        pred_g, norm_meta = normalize_pred_grid_for_display(pred_raw)
        if adaptive_threshold:
            thr_metric, adapt_meta = adaptive_threshold_day_grid(
                real_g,
                pred_g,
                float(pred_threshold),
                min_recall=float(adaptive_min_recall),
                min_precision_floor=float(adaptive_min_precision),
                fp_penalty=float(adaptive_fp_penalty),
            )
        else:
            thr_metric = float(pred_threshold)
            pb0 = (pred_g >= thr_metric).astype(np.float32)
            m0 = binary_metrics(real_g, pb0)
            adapt_meta = {"adaptive_model": "none", "mode": "fixed", "chosen": thr_metric, **m0}
        pred_bin = (pred_g >= thr_metric).astype(np.float32)

        m_bin = binary_metrics(real_g, pred_bin)
        n_focos = int(counts.get(day, 0))
        day_str = str(day)
        m_bin["n_focos"] = n_focos
        m_bin["date"] = day_str
        m_bin["n_cube_frames_that_day"] = int(n_frames)
        m_bin.update({f"pred_{k}": v for k, v in norm_meta.items()})
        m_bin["pred_threshold_base"] = float(pred_threshold)
        m_bin["pred_threshold"] = float(thr_metric)
        m_bin["adaptive_threshold_meta"] = adapt_meta

        from src.compare_ceara_maps import (
            display_positive_mask,
            map_points_to_json_fields,
            pred_points_from_display_mask,
            real_foci_points_day,
            save_ceara_folium_map,
            save_ceara_map_png,
        )

        mask_vis, thr_vis, disp_mode = display_positive_mask(pred_g, float(thr_metric))
        m_bin["pred_display_threshold"] = float(thr_vis)
        m_bin["pred_display_mode"] = disp_mode

        png = out / f"{file_prefix}_real_vs_pred_{day_str}.png"
        save_comparison_figure(
            real_g,
            pred_g,
            thr_metric,
            day_str,
            m_bin,
            png,
            display_mask=mask_vis,
            display_thr=thr_vis,
            display_mode=disp_mode,
        )
        m_bin["figure"] = str(png)

        real_pts = real_foci_points_day(dfx, day)
        pred_pts = pred_points_from_display_mask(pred_g, n_lat, n_lon, mask_vis)
        html_path = out / f"{file_prefix}_ceara_map_{day_str}.html"
        png_map = out / f"{file_prefix}_ceara_map_{day_str}.png"
        save_ceara_folium_map(
            html_path,
            f"Ceará — {file_prefix} — {day_str}",
            real_pts,
            pred_pts,
            thr_metric,
            display_threshold=thr_vis,
        )
        save_ceara_map_png(
            png_map,
            f"Ceará — {day_str} — Reais vs previsto ({file_prefix})",
            real_pts,
            pred_pts,
            m_bin,
            pred_raster=pred_g,
            display_thr=thr_vis,
        )
        m_bin["map_html"] = str(html_path)
        m_bin["map_png"] = str(png_map)
        m_bin.update(map_points_to_json_fields(real_pts, pred_pts))
        rows.append(m_bin)

    summary = {
        "technique": "GOES-16 proxy cube + Isolation Forest (unsupervised) vs daily focal grid",
        "caveat": "O cubo GOES-proxy é derivado dos próprios focos; métricas não são validação independente de ABI L1b.",
        "year_filter": int(year) if year is not None else None,
        "cube_time_span": {
            "start": str(pd.Timestamp(times[0])) if len(times) else None,
            "end": str(pd.Timestamp(times[-1])) if len(times) else None,
        },
        "pred_threshold": float(pred_threshold),
        "adaptive_threshold": {
            "enabled": bool(adaptive_threshold),
            "min_recall": float(adaptive_min_recall),
            "min_precision_floor": float(adaptive_min_precision),
            "fp_penalty": float(adaptive_fp_penalty),
        },
        "grid_shape": [int(n_lat), int(n_lon)],
        "config": {
            "grid_resolution": cfg.grid_resolution,
            "frame_minutes": cfg.frame_minutes,
            "max_days_history": cfg.max_days_history,
            "netcdf": netcdf_path,
        },
        "days_evaluated": len(rows),
        "metrics_by_day": rows,
    }
    from src.compare_ceara_maps import save_all_metrics_outputs

    return save_all_metrics_outputs(out, rows, summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara dias com focos reais vs predição GOES não supervisionada e salva PNGs."
    )
    parser.add_argument("--csv", required=True, help="CSV com datetime (ou data_hora) e lat/lon")
    parser.add_argument("--out", default="data/goes_unsup_compare", help="Diretório de saída (PNGs + JSON)")
    parser.add_argument(
        "--max-days",
        type=int,
        default=12,
        help="Máximo de dias (por nº de focos). 0 = todos os dias com foco dentro do cubo.",
    )
    parser.add_argument("--year", type=int, default=None, help="Filtra focos por ano civil (ex.: 2024)")
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=0.35,
        help="Limiar base (piso); com adaptativo, o limiar efetivo por dia pode subir para cortar FP.",
    )
    parser.add_argument(
        "--no-adaptive-threshold",
        action="store_true",
        help="Desliga o limiar adaptativo por dia (usa só --pred-threshold).",
    )
    parser.add_argument("--adapt-min-recall", type=float, default=0.12, help="Recall mínimo no limiar adaptativo")
    parser.add_argument(
        "--adapt-min-precision",
        type=float,
        default=0.06,
        help="Piso de precisão (só candidatos com TP>0) no limiar adaptativo",
    )
    parser.add_argument("--adapt-fp-penalty", type=float, default=0.42, help="Peso da taxa de FP no objetivo adaptativo")
    parser.add_argument("--grid-resolution", type=float, default=0.1, help="Resolução do cubo proxy (graus)")
    parser.add_argument("--max-days-history", type=int, default=30, help="Janela de histórico no cubo (dias)")
    parser.add_argument("--goes-proxy-netcdf", type=str, default="", help="NetCDF opcional do cubo GOES-proxy")
    parser.add_argument("--frame-minutes", type=int, default=5, help="Resolução temporal do cubo (minutos)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    cfg = GOESUnsupervisedConfig(
        grid_resolution=float(args.grid_resolution),
        max_days_history=int(args.max_days_history),
        frame_minutes=int(args.frame_minutes),
    )
    summary = compare_fire_days_and_save_figures(
        df,
        output_dir=args.out,
        cfg=cfg,
        netcdf_path=args.goes_proxy_netcdf or None,
        max_days=int(args.max_days),
        pred_threshold=float(args.pred_threshold),
        year=args.year,
        adaptive_threshold=not bool(args.no_adaptive_threshold),
        adaptive_min_recall=float(args.adapt_min_recall),
        adaptive_min_precision=float(args.adapt_min_precision),
        adaptive_fp_penalty=float(args.adapt_fp_penalty),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics_by_day"}, ensure_ascii=False, indent=2))
    print(f"Figuras e JSON em: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
