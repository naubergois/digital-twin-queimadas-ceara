"""CLI para baixar dados reais de focos de queimadas por ano.

O script usa as rotinas de coleta já existentes no projeto e grava:
- CSVs mensais em `data/`
- CSV consolidado anual em `data/focos_CE_<ano>_completo.csv`

Exemplos:
    .venv/bin/python3 -m src.download_real_years --years 2024 2025 2026
    .venv/bin/python3 -m src.download_real_years --start-year 2024 --end-year 2026
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from src.fire_data import download_year_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa dados reais anuais de focos de queimadas para o Ceará."
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Lista de anos a baixar, por exemplo: --years 2024 2025 2026",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Ano inicial para baixar um intervalo contínuo.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Ano final para baixar um intervalo contínuo.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Diretório de saída dos CSVs (padrão: data).",
    )
    return parser.parse_args()


def resolve_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        years = list(dict.fromkeys(args.years))
        return sorted(years)

    if args.start_year is not None and args.end_year is not None:
        if args.start_year > args.end_year:
            raise ValueError("--start-year precisa ser menor ou igual a --end-year")
        return list(range(args.start_year, args.end_year + 1))

    raise ValueError("Informe --years ou o par --start-year/--end-year")


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def download_years(years: Iterable[int], output_dir: str) -> int:
    failures = 0
    for year in years:
        print(f"\n=== Baixando ano {year} ===")
        df_year = download_year_data(year=year, output_dir=output_dir)

        if df_year.empty:
            failures += 1
            print(
                f"[ERRO] Nenhum dado real encontrado para {year}. "
                "A fonte INPE pode estar indisponível para esse período."
            )
            continue

        print(f"[OK] {year}: {len(df_year)} focos baixados com sucesso.")

    return failures


def main() -> int:
    args = parse_args()

    try:
        years = resolve_years(args)
    except ValueError as exc:
        print(f"[USO] {exc}")
        return 2

    output_dir = ensure_output_dir(args.output_dir)
    failures = download_years(years, output_dir)

    if failures:
        print(f"\n[FINAL] Concluído com {failures} ano(s) sem dados reais.")
        return 1

    print("\n[FINAL] Download concluído com sucesso para todos os anos solicitados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())