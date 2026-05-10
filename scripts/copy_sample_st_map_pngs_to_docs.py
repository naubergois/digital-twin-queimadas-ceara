#!/usr/bin/env python3
"""
Copia figuras de mapa / grade da comparação ST-HyperNet para docs/screenshots/,
para o README mostrar exemplos com achados (reais vs previsto) sem depender de data/ no git.

Uso (na raiz do repo):
  python scripts/copy_sample_st_map_pngs_to_docs.py
  python scripts/copy_sample_st_map_pngs_to_docs.py --date 2024-11-30 --metrics-dir data/st_hypernet_2024_all_fire_days
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "docs" / "screenshots"


def main() -> int:
    p = argparse.ArgumentParser(description="Copia PNGs de mapa ST-HyperNet para docs/screenshots/")
    p.add_argument(
        "--metrics-dir",
        default="data/st_hypernet_2024_all_fire_days",
        help="Pasta com st_hypernet_ceara_map_*.png e st_hypernet_real_vs_pred_*.png",
    )
    p.add_argument("--date", default="2024-08-09", help="Dia YYYY-MM-DD (ficheiros prefixo st_hypernet_)")
    p.add_argument("--prefix", default="st_hypernet", help="Prefixo dos ficheiros gerados por compare_st_hypernet_days")
    args = p.parse_args()

    src_dir = (REPO / args.metrics_dir).resolve()
    if not src_dir.is_dir():
        print(f"Pasta inexistente: {src_dir}", file=sys.stderr)
        return 1

    day = str(args.date).strip()
    prefix = str(args.prefix).strip()
    pairs = [
        (src_dir / f"{prefix}_ceara_map_{day}.png", DEST / "mapa-ceara-reais-vs-previsto.png"),
        (src_dir / f"{prefix}_real_vs_pred_{day}.png", DEST / "grade-real-vs-previsto.png"),
    ]
    DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, dst in pairs:
        if not src.is_file():
            print(f"Aviso: não encontrado {src}", file=sys.stderr)
            continue
        shutil.copy2(src, dst)
        print(dst)
        copied += 1
    if copied == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
