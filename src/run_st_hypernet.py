#!/usr/bin/env python3
"""Treina ST-HyperNet em cubo GOES-proxy e exporta relatório + artefato .pt."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.st_hypernet import (
    STHyperNetConfig,
    export_public_st_report,
    run_st_hypernet_pipeline,
    save_st_hypernet_artifact,
    write_st_hypernet_best_params_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description="ST-HyperNet — treino self-supervised em cubo GOES-proxy")
    p.add_argument("--csv", required=True, help="CSV com datetime, lat, lon")
    p.add_argument("--out-dir", default="data/st_hypernet", help="Diretório de saída")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="cpu", help="cpu ou cuda")
    p.add_argument("--grid-resolution", type=float, default=0.5)
    p.add_argument("--frame-minutes", type=int, default=60)
    p.add_argument(
        "--max-days-history",
        type=int,
        default=0,
        help="0 = todo o período do CSV; >0 = últimos N dias até o último foco.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)

    cfg = STHyperNetConfig(
        epochs=int(args.epochs),
        device=str(args.device),
        grid_resolution=float(args.grid_resolution),
        frame_minutes=int(args.frame_minutes),
        max_days_history=int(args.max_days_history),
    )
    result = run_st_hypernet_pipeline(df, cfg=cfg)
    save_st_hypernet_artifact(result, os.path.join(out, "st_hypernet.pt"))
    write_st_hypernet_best_params_json(
        result,
        os.path.join(out, "st_hypernet_best_params.json"),
        extras={"source": "run_st_hypernet", "out_dir": out},
    )

    public = export_public_st_report(result)
    with open(os.path.join(out, "st_hypernet_report.json"), "w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, ensure_ascii=False)

    print(json.dumps(public, indent=2, ensure_ascii=False))
    print(f"Artefatos em: {out}")


if __name__ == "__main__":
    main()
