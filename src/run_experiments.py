#!/usr/bin/env python3
"""Executa e compara experimentos de tecnicas para o Digital Twin de queimadas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml_digital_twin import FireMLDigitalTwin, MLTwinConfig


DEFAULT_DATASET = "data/focos_CE_GOES16_2024.csv"
DEFAULT_OUTPUT_DIR = "data/experiments"

FIRES_JSON_NAME = "dataset_fires_detail.json"
FIRES_CSV_NAME = "dataset_fires_detail.csv"


def _fire_records_jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte o DataFrame de focos em registros JSON (datas ISO, nulls, índice estável)."""
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[col]):
            d[col] = d[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    records: list[dict[str, Any]] = json.loads(d.to_json(orient="records", date_format="iso"))
    for i, rec in enumerate(records):
        rec["_row_index"] = i
        for k, v in list(rec.items()):
            if isinstance(v, float) and np.isnan(v):
                rec[k] = None
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
    return records


def _build_input_fire_dataset_summary(df: pd.DataFrame, data_file: Path) -> dict[str, Any]:
    """Resumo da base de focos: colunas, contagem por fonte (`source`) e por satélite, caminho."""
    cols = list(df.columns)
    summary: dict[str, Any] = {
        "dataset_path": str(data_file),
        "dataset_path_resolved": str(data_file.resolve()),
        "n_fires": int(len(df)),
        "columns": cols,
        "column_descriptions": {
            "lat": "Latitude (graus)",
            "lon": "Longitude (graus)",
            "datetime": "Data/hora do detecção",
            "satellite": "Satélite ou sensor (ex.: GOES-16)",
            "source": "Procedência do registro (ex.: INPE API, CSV anual, FIRMS)",
            "municipio": "Município (quando informado pela fonte)",
            "estado": "Estado",
            "bioma": "Bioma",
            "frp": "Fire Radiative Power (quando disponível)",
            "risco_fogo": "Índice ou classificação de risco (fonte INPE)",
            "pais": "País",
        },
        "fires_detail_json": FIRES_JSON_NAME,
        "fires_detail_csv": FIRES_CSV_NAME,
    }
    if "source" in df.columns:
        summary["counts_by_source"] = df["source"].fillna("(sem fonte)").astype(str).value_counts().to_dict()
    else:
        summary["counts_by_source"] = {}
    if "satellite" in df.columns:
        summary["counts_by_satellite"] = (
            df["satellite"].fillna("(sem satélite)").astype(str).value_counts().to_dict()
        )
    else:
        summary["counts_by_satellite"] = {}
    if "municipio" in df.columns:
        summary["top_municipios"] = (
            df["municipio"].fillna("").astype(str).str.strip().replace("", "(sem município)").value_counts().head(15).to_dict()
        )
    return summary


def _write_fires_dataset_artifacts(out_dir: Path, df: pd.DataFrame) -> None:
    """Grava JSON com cada queimada (todas as colunas) + CSV espelho no diretório do experimento."""
    detail = {
        "description": "Cada registro é um foco de calor usado como entrada desta rodada de experimentos.",
        "fires": _fire_records_jsonable(df),
    }
    (out_dir / FIRES_JSON_NAME).write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_dir / FIRES_CSV_NAME, index=False)


def _technique_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "baseline_strict_legacy",
            "description": "Baseline estrito sem custo sensivel e sem recencia",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 3,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 6,
                "use_cost_sensitive": False,
                "recency_weight_power": 0.0,
            },
        },
        {
            "name": "strict_cost_sensitive",
            "description": "Aprendizado cost-sensitive para desbalanceamento extremo",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 3,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 6,
                "use_cost_sensitive": True,
                "recency_weight_power": 0.8,
            },
        },
        {
            "name": "strict_high_positive_weight",
            "description": "Cost-sensitive com peso positivo fixo para elevar recall",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 3,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 6,
                "use_cost_sensitive": True,
                "positive_class_weight": 24.0,
                "recency_weight_power": 0.8,
            },
        },
        {
            "name": "strict_long_memory",
            "description": "Janela temporal maior para detectar padroes persistentes",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 7,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "strict_spatial_richer",
            "description": "Grade mais fina e features espaciais enriquecidas",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.2,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "strict_tolerant_optimization",
            "description": "Otimizacao por F1 tolerante para reduzir erro espacial",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1_tolerant",
                "tolerant_radius_cells": 1,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "strict_no_hnm",
            "description": "Teste sem hard-negative mining",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": True,
                "use_hard_negative_mining": False,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "strict_no_deep",
            "description": "Ablacao removendo a rede temporal",
            "config": {
                "mode": "strict_cell",
                "grid_resolution": 0.25,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1",
                "tolerant_radius_cells": 0,
                "use_deep_temporal": False,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "operational_day_focus",
            "description": "Modo operacional priorizando deteccao diaria",
            "config": {
                "mode": "operational",
                "grid_resolution": 0.25,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1_day",
                "tolerant_radius_cells": 2,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
        {
            "name": "operational_tolerant_focus",
            "description": "Modo operacional com foco em acerto espacial tolerante",
            "config": {
                "mode": "operational",
                "grid_resolution": 0.25,
                "lookback_days": 5,
                "auto_calibrate": True,
                "optimize_metric": "f1_tolerant",
                "tolerant_radius_cells": 2,
                "use_deep_temporal": True,
                "use_hard_negative_mining": True,
                "hnm_neg_pos_ratio": 8,
                "use_cost_sensitive": True,
                "recency_weight_power": 1.0,
            },
        },
    ]


def _extract_row(exp: dict[str, Any], result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    ml = result.get("ml_metrics", {})
    twin = result.get("twin_metrics", {})
    best = result.get("model_selection", {}).get("best_model", {})
    cfg = result.get("config", {})

    return {
        "experiment": exp["name"],
        "description": exp["description"],
        "status": "ok",
        "elapsed_sec": round(elapsed, 2),
        "best_model": best.get("model", ""),
        "mode": cfg.get("mode", ""),
        "grid_resolution": cfg.get("grid_resolution", ""),
        "lookback_days": cfg.get("lookback_days", ""),
        "optimize_metric": cfg.get("optimize_metric", ""),
        "ml_precision": ml.get("precision", float("nan")),
        "ml_recall": ml.get("recall", float("nan")),
        "ml_f1": ml.get("f1", float("nan")),
        "ml_f1_tolerant": ml.get("f1_tolerant", float("nan")),
        "ml_f1_day": ml.get("f1_day", float("nan")),
        "twin_f1": twin.get("f1", float("nan")),
        "twin_f1_tolerant": twin.get("f1_tolerant", float("nan")),
        "twin_f1_day": twin.get("f1_day", float("nan")),
        "strict_score": ml.get("f1", float("nan")),
        "operational_score": (
            0.55 * float(ml.get("f1_tolerant", 0.0))
            + 0.45 * float(ml.get("f1_day", 0.0))
        ),
    }


def _extract_error_row(exp: dict[str, Any], err: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "experiment": exp["name"],
        "description": exp["description"],
        "status": "error",
        "elapsed_sec": round(elapsed, 2),
        "error": str(err),
    }


def _write_markdown_report(df: pd.DataFrame, output_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Experiment Comparison")
    lines.append("")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        lines.append("Nenhum experimento concluido com sucesso.")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        return

    best_strict = ok.sort_values("strict_score", ascending=False).head(1).iloc[0]
    best_oper = ok.sort_values("operational_score", ascending=False).head(1).iloc[0]

    lines.append("## Best Results")
    lines.append("")
    lines.append(
        f"- Best strict F1: **{best_strict['experiment']}** (F1={best_strict['ml_f1']:.4f}, model={best_strict['best_model']})"
    )
    lines.append(
        f"- Best operational score: **{best_oper['experiment']}** (score={best_oper['operational_score']:.4f}, F1_tol={best_oper['ml_f1_tolerant']:.4f}, F1_day={best_oper['ml_f1_day']:.4f})"
    )
    lines.append("")

    lines.append("## Ranking by Strict F1")
    lines.append("")
    strict_rank = ok.sort_values("strict_score", ascending=False)
    for i, (_, row) in enumerate(strict_rank.iterrows(), start=1):
        lines.append(
            f"{i}. {row['experiment']} - F1={row['ml_f1']:.4f}, P={row['ml_precision']:.4f}, R={row['ml_recall']:.4f}, model={row['best_model']}"
        )
    lines.append("")

    lines.append("## Ranking by Operational Score")
    lines.append("")
    oper_rank = ok.sort_values("operational_score", ascending=False)
    for i, (_, row) in enumerate(oper_rank.iterrows(), start=1):
        lines.append(
            f"{i}. {row['experiment']} - Score={row['operational_score']:.4f}, F1_tol={row['ml_f1_tolerant']:.4f}, F1_day={row['ml_f1_day']:.4f}"
        )
    lines.append("")

    lines.append("## Notes on Tested Techniques")
    lines.append("")
    lines.append("- Cost-sensitive learning: aumenta o peso de classes positivas em cenarios desbalanceados.")
    lines.append("- Recency weighting: prioriza amostras recentes para adaptacao temporal.")
    lines.append("- Hard-negative mining: foca negativos dificeis para reduzir falsos positivos.")
    lines.append("- Soft voting ensemble: combina arvores e modelo linear para robustez.")
    lines.append("- Metric-aware optimization: alterna objetivo entre F1 estrito, F1 tolerante e F1 diario.")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def _write_protocol(protocol_md: Path, dataset_path: str) -> None:
    text = """# Experiment Protocol

## Objective
Improve wildfire focus detection metrics in Ceara and compare multiple techniques under a reproducible setup.

## Dataset
- Input file: {dataset}
- Expected columns: datetime, lat, lon (optional municipio)

## Procedure
1. Keep a fixed temporal split (train/validation/test) inside the same validation pipeline.
2. Run all experiments with deterministic seeds.
3. Save one JSON per run and aggregate summary in CSV and JSON.
4. Compare two views:
   - Strict cell-level quality (ml_f1)
   - Operational quality (weighted combination of ml_f1_tolerant and ml_f1_day)

## Tested Technique Families
- Cost-sensitive learning for class imbalance.
- Recency-aware sample weighting.
- Hard-negative mining.
- Ensemble model selection with multiple candidate classifiers.
- Metric-aware optimization (strict and tolerant criteria).

## Output Artifacts
- data/experiments/all_experiments_summary.csv
- data/experiments/all_experiments_full.json (inclui bloco `input_fire_dataset` + lista `experiments`)
- data/experiments/dataset_fires_detail.json (cada foco com todas as colunas do CSV, incl. `source`)
- data/experiments/dataset_fires_detail.csv (espelho tabular da mesma base)
- data/experiments/runs/*.json (métricas; ``best_model_params`` vai em ``*_best_params.json`` ao lado)
- data/experiments/runs/*_best_params.json (hiperparâmetros MLTwinConfig + limiares + params do classificador escolhido)
- EXPERIMENTS.md

## Diretrizes humanas (manutenção)
Ver o guia em **GUIA_EXPERIMENTOS_E_PARAMETROS.md** (raiz do repositório): salvar e organizar experimentos, e sempre persistir os melhores parâmetros por modelo.

## Re-run Command
python -m src.run_experiments --dataset {dataset}
""".format(dataset=dataset_path)
    protocol_md.write_text(text, encoding="utf-8")


def run(dataset_path: str, output_dir: str) -> None:
    data_file = Path(dataset_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {data_file}")

    out_dir = Path(output_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    _write_fires_dataset_artifacts(out_dir, df)
    input_fire_dataset = _build_input_fire_dataset_summary(df, data_file)

    all_rows: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []

    for exp in _technique_catalog():
        print(f"[RUN] {exp['name']}")
        cfg = MLTwinConfig(**exp["config"])
        start = time.perf_counter()
        try:
            result = FireMLDigitalTwin(cfg).validate_with_real_data(df)
            result["input_fire_dataset"] = input_fire_dataset
            elapsed = time.perf_counter() - start
            row = _extract_row(exp, result, elapsed)
            all_rows.append(row)

            run_snapshot = dict(result)
            best_params_blob = run_snapshot.pop("best_model_params", None)
            full_results.append(
                {
                    "experiment": exp["name"],
                    "description": exp["description"],
                    "elapsed_sec": round(elapsed, 2),
                    "result": run_snapshot,
                    "best_params_json": f"{exp['name']}_best_params.json" if best_params_blob else None,
                }
            )

            run_file = runs_dir / f"{exp['name']}.json"
            run_file.write_text(json.dumps(run_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            if best_params_blob is not None:
                (runs_dir / f"{exp['name']}_best_params.json").write_text(
                    json.dumps(best_params_blob, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(
                f"  -> model={row['best_model']} f1={row['ml_f1']:.4f} "
                f"f1_tol={row['ml_f1_tolerant']:.4f} f1_day={row['ml_f1_day']:.4f}"
            )
        except Exception as err:
            elapsed = time.perf_counter() - start
            all_rows.append(_extract_error_row(exp, err, elapsed))
            full_results.append(
                {
                    "experiment": exp["name"],
                    "description": exp["description"],
                    "elapsed_sec": round(elapsed, 2),
                    "error": str(err),
                }
            )
            print(f"  -> ERROR: {err}")

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(out_dir / "all_experiments_summary.csv", index=False)
    full_bundle = {
        "input_fire_dataset": input_fire_dataset,
        "experiments": full_results,
    }
    (out_dir / "all_experiments_full.json").write_text(
        json.dumps(full_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_markdown_report(summary_df, Path("EXPERIMENTS.md"))
    _write_protocol(Path("EXPERIMENT_PROTOCOL.md"), str(data_file))

    print("\nArquivos gerados:")
    print(f"- {out_dir / 'all_experiments_summary.csv'}")
    print(f"- {out_dir / 'all_experiments_full.json'}")
    print(f"- {out_dir / FIRES_JSON_NAME} (focos com colunas completas, ex. source)")
    print(f"- {out_dir / FIRES_CSV_NAME}")
    print(f"- {runs_dir} (inclui <experimento>_best_params.json por execução bem-sucedida)")
    print("- EXPERIMENTS.md")
    print("- EXPERIMENT_PROTOCOL.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rodar comparacao de tecnicas ML do Digital Twin")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="CSV com focos de queimadas")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Diretorio para resultados")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dataset_path=args.dataset, output_dir=args.output_dir)
