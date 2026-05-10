#!/usr/bin/env python3
"""
Gera PNGs em docs/screenshots/ para o README: rankings de experimentos ML e resumo ST-HyperNet.

Usa ficheiros versionados em docs/fixtures/ (não depende de data/ no git).

  python scripts/generate_readme_experiment_figures.py
  python scripts/generate_readme_experiment_figures.py --experiments-csv data/experiments/all_experiments_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_MPL = REPO / ".mplconfig"
_MPL.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = REPO / "docs" / "screenshots"
DEFAULT_CSV = REPO / "docs" / "fixtures" / "all_experiments_summary_sample.csv"
DEFAULT_ST_JSON = REPO / "docs" / "fixtures" / "metrics_aggregate_st_hypernet_sample.json"


def _short_label(name: str, max_len: int = 28) -> str:
    s = str(name)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def plot_experiment_rankings(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("CSV sem linhas status=ok", file=sys.stderr)
        return

    ok = ok.sort_values("ml_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(ok))
    ax.barh(y, ok["ml_f1"].astype(float), color="#2c5282", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(x) for x in ok["experiment"]], fontsize=9)
    ax.set_xlabel("F1 (célula estrita, ML)")
    ax.set_title("Comparação de experimentos — ranking por F1 estrito (ML)")
    ax.set_xlim(0, max(0.35, float(ok["ml_f1"].max()) * 1.15))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "experimentos-ranking-f1-ml.png", dpi=144, bbox_inches="tight")
    plt.close(fig)

    ok2 = df[df["status"] == "ok"].copy().sort_values("operational_score", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(10, 5.2))
    y2 = np.arange(len(ok2))
    ax2.barh(y2, ok2["operational_score"].astype(float), color="#276749", height=0.65)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([_short_label(x) for x in ok2["experiment"]], fontsize=9)
    ax2.set_xlabel("Score operacional (0.55·F1_tol + 0.45·F1_dia)")
    ax2.set_title("Comparação de experimentos — ranking por score operacional")
    ax2.set_xlim(0, max(0.8, float(ok2["operational_score"].max()) * 1.08))
    ax2.grid(axis="x", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT / "experimentos-ranking-score-operacional.png", dpi=144, bbox_inches="tight")
    plt.close(fig2)


def plot_st_aggregate(path: Path) -> None:
    if not path.is_file():
        print(f"Aviso: {path} inexistente — skip ST", file=sys.stderr)
        return
    agg = json.loads(path.read_text(encoding="utf-8"))
    labels = ["IoU médio", "Precisão média", "Recall médio", "Micro prec.", "Micro recall"]
    vals = [
        float(agg.get("mean_iou", 0)),
        float(agg.get("mean_precision", 0)),
        float(agg.get("mean_recall", 0)),
        float(agg.get("micro_precision", 0)),
        float(agg.get("micro_recall", 0)),
    ]
    colors = ["#4a5568", "#2b6cb0", "#b7791f", "#276749", "#9b2c2c"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor [0–1]")
    n = int(agg.get("n_days", 0))
    ax.set_title(f"Comparação ST-HyperNet (agregado, exemplo — {n} dias)")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, min(1.0, v + 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "comparacao-st-hypernet-agregado.png", dpi=144, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--st-aggregate-json", type=Path, default=DEFAULT_ST_JSON)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = args.experiments_csv
    if not csv_path.is_file():
        alt = REPO / "data" / "experiments" / "all_experiments_summary.csv"
        if alt.is_file():
            csv_path = alt
        else:
            print(f"CSV não encontrado: {args.experiments_csv}", file=sys.stderr)
            return 1

    df = pd.read_csv(csv_path)
    plot_experiment_rankings(df)
    plot_st_aggregate(args.st_aggregate_json)
    print("OK:", OUT / "experimentos-ranking-f1-ml.png")
    print("OK:", OUT / "experimentos-ranking-score-operacional.png")
    print("OK:", OUT / "comparacao-st-hypernet-agregado.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
