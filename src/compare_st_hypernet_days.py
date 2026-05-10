"""
Compara dias com focos reais vs cubo de score ST-HyperNet e salva PNGs + JSON.

Reutiliza grades diárias e figuras de `compare_goes_unsupervised_days`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.compare_goes_unsupervised_days import (
    _prepare_foci_df,
    adaptive_threshold_day_grid,
    binary_metrics,
    daily_pred_grid,
    daily_real_grid,
    fire_days_in_cube_span,
    normalize_pred_grid_for_display,
    save_comparison_figure,
)
from src.st_hypernet import (
    STHyperNetConfig,
    run_st_hypernet_pipeline,
    save_st_hypernet_artifact,
    write_st_hypernet_best_params_json,
)


def compare_st_hypernet_fire_days_and_save_figures(
    df: pd.DataFrame,
    output_dir: str | Path,
    cfg: STHyperNetConfig | None = None,
    max_days: int = 12,
    pred_threshold: float = 0.35,
    file_prefix: str = "st_hypernet",
    year: int | None = None,
    adaptive_threshold: bool = True,
    adaptive_min_recall: float = 0.12,
    adaptive_min_precision: float = 0.06,
    adaptive_fp_penalty: float = 0.42,
) -> Dict[str, Any]:
    """
    Treina ST-HyperNet no cubo GOES-proxy, compara por dia civil com focos na grade
    e grava PNGs ``st_hypernet_real_vs_pred_<data>.png`` + ``metrics_by_day.json``.
    """
    cfg = cfg or STHyperNetConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dfx = _prepare_foci_df(df)
    if dfx.empty:
        raise ValueError("Sem focos válidos no bounding box do Ceará.")
    if year is not None:
        dfx = dfx[pd.to_datetime(dfx["datetime"]).dt.year == int(year)].copy()
    if dfx.empty:
        raise ValueError(f"Sem focos após filtro year={year}.")

    result = run_st_hypernet_pipeline(dfx, cfg=cfg)
    save_st_hypernet_artifact(result, out / f"{file_prefix}_best_model.pt")
    cube = result["_score_cube"]
    times = result["_times"]
    T, n_lat, n_lon = cube.shape

    counts = dfx.groupby("date").size().sort_values(ascending=False)
    fire_days = fire_days_in_cube_span(counts, times, max_days)
    if not fire_days:
        raise ValueError(
            "Nenhum dia com focos intersecta o eixo temporal do cubo ST. "
            "Aumente --max-days-history ou use 0 (cubo completo; pode ser lento)."
        )

    rows = []
    for day in fire_days:
        real_g = daily_real_grid(dfx, day, n_lat, n_lon)
        pred_raw, n_frames = daily_pred_grid(cube, times, day)
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
        "technique": "ST-HyperNet score cube vs daily focal grid (GOES-proxy)",
        "caveat": "Cubo GOES-proxy derivado dos focos; métricas exploratórias.",
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
        "days_evaluated": len(rows),
        "metrics_by_day": rows,
        "train_loss_last": (result.get("train_meta") or {}).get("loss_last"),
    }
    from src.compare_ceara_maps import save_all_metrics_outputs

    summary = save_all_metrics_outputs(out, rows, summary)
    write_st_hypernet_best_params_json(
        result,
        out / f"{file_prefix}_best_params.json",
        extras={
            "pred_threshold": float(pred_threshold),
            "adaptive_threshold": summary.get("adaptive_threshold"),
            "year_filter": int(year) if year is not None else None,
            "file_prefix": file_prefix,
            "days_evaluated": summary.get("days_evaluated"),
            "metrics_aggregate": summary.get("metrics_aggregate"),
            "cube_time_span": summary.get("cube_time_span"),
        },
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Compara ST-HyperNet vs focos reais por dia (PNGs)")
    p.add_argument("--csv", required=True, help="CSV com datetime, lat, lon")
    p.add_argument("--out", default="data/st_hypernet_compare", help="Diretório de saída")
    p.add_argument(
        "--max-days",
        type=int,
        default=12,
        help="Máximo de dias (por nº de focos). 0 = todos os dias com foco dentro do cubo.",
    )
    p.add_argument("--year", type=int, default=None, help="Filtra focos por ano civil (ex.: 2024)")
    p.add_argument(
        "--pred-threshold",
        type=float,
        default=0.22,
        help="Limiar base (piso); com adaptativo o limiar efetivo por dia pode subir para reduzir FP.",
    )
    p.add_argument(
        "--no-adaptive-threshold",
        action="store_true",
        help="Desliga o limiar adaptativo por dia.",
    )
    p.add_argument("--adapt-min-recall", type=float, default=0.12)
    p.add_argument("--adapt-min-precision", type=float, default=0.06)
    p.add_argument("--adapt-fp-penalty", type=float, default=0.42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--grid-resolution", type=float, default=0.5)
    p.add_argument("--frame-minutes", type=int, default=60)
    p.add_argument(
        "--max-days-history",
        type=int,
        default=0,
        help="Janela do cubo: últimos N dias a partir do último foco. 0 = usar todo o período do CSV (recomendado p/ 2024 completo).",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--inference-stride",
        type=int,
        default=1,
        help="Stride espacial na inferência ST-HyperNet (2 ou 3 acelera o ano inteiro com leve perda de resolução).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    cfg = STHyperNetConfig(
        epochs=int(args.epochs),
        grid_resolution=float(args.grid_resolution),
        frame_minutes=int(args.frame_minutes),
        max_days_history=int(args.max_days_history),
        device=str(args.device),
        max_patches_per_epoch=4096,
        inference_stride=max(1, int(args.inference_stride)),
    )
    summary = compare_st_hypernet_fire_days_and_save_figures(
        df,
        output_dir=args.out,
        cfg=cfg,
        max_days=int(args.max_days),
        pred_threshold=float(args.pred_threshold),
        year=args.year,
        adaptive_threshold=not bool(args.no_adaptive_threshold),
        adaptive_min_recall=float(args.adapt_min_recall),
        adaptive_min_precision=float(args.adapt_min_precision),
        adaptive_fp_penalty=float(args.adapt_fp_penalty),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics_by_day"}, ensure_ascii=False, indent=2))
    print(f"Saída: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
