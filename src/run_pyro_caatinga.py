#!/usr/bin/env python3
"""Runner CLI para o PYRO-Caatinga MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pyro_caatinga import PyroCaatingaConfig, run_pyro_caatinga_mvp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o PYRO-Caatinga MVP")
    parser.add_argument("--goes-csv", required=True, help="CSV principal com datetime/lat/lon")
    parser.add_argument("--viirs-csv", default="", help="CSV VIIRS para destilacao (opcional)")
    parser.add_argument("--output-dir", default="data/pyro_caatinga", help="Diretorio de saida")
    parser.add_argument("--grid", type=float, default=0.1, help="Resolucao da grade")
    parser.add_argument("--frame-min", type=int, default=5, help="Janela temporal em minutos")
    parser.add_argument("--max-days", type=int, default=7, help="Historico maximo de dias para montar cubo")
    parser.add_argument("--ewma", type=float, default=0.05, help="Lambda da climatologia EWMA")
    parser.add_argument("--tau", type=float, default=0.015, help="Limiar de incerteza do loop twin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goes_df = pd.read_csv(args.goes_csv)
    viirs_df = pd.read_csv(args.viirs_csv) if args.viirs_csv else None

    cfg = PyroCaatingaConfig(
        grid_resolution=args.grid,
        frame_minutes=args.frame_min,
        max_days_history=args.max_days,
        ewma_lambda=args.ewma,
        uncertainty_tau=args.tau,
    )

    report = run_pyro_caatinga_mvp(
        goes_df=goes_df,
        viirs_df=viirs_df,
        output_dir=args.output_dir,
        cfg=cfg,
    )

    print("PYRO-Caatinga MVP concluido.")
    print(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))
    print(f"Relatorio: {Path(args.output_dir) / 'pyro_caatinga_report.json'}")


if __name__ == "__main__":
    main()
