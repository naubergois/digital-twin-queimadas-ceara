#!/usr/bin/env python3
"""
Regenera o PNG «mapa Ceará reais vs previsto» para docs/screenshots/ a partir de
`metrics_by_day.json` (pontos serializados), com fundo de satélite real (ESRI via contextily).

Requer rede na primeira vez (cache de tiles). Uso:

  python scripts/render_readme_ceara_map_from_metrics.py
  python scripts/render_readme_ceara_map_from_metrics.py --date 2024-11-30 \\
      --metrics-dir data/st_hypernet_2024_all_fire_days \\
      --out docs/screenshots/mapa-ceara-reais-vs-previsto.png
  python scripts/render_readme_ceara_map_from_metrics.py --demo --out docs/screenshots/mapa-ceara-reais-vs-previsto.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    if "MPLCONFIGDIR" not in os.environ:
        mpl = REPO / ".mplconfig"
        mpl.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")

    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", default="data/st_hypernet_2024_all_fire_days")
    p.add_argument("--date", default="2024-08-09")
    p.add_argument("--out", type=Path, default=REPO / "docs" / "screenshots" / "mapa-ceara-reais-vs-previsto.png")
    p.add_argument("--no-satellite", action="store_true", help="Desliga fundo ESRI (só para teste offline)")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Ignora pontos do JSON: usa pontos de exemplo no Ceará (só para validar basemap satélite; rerode compare para dados reais).",
    )
    args = p.parse_args()

    sys.path.insert(0, str(REPO))
    base = (REPO / args.metrics_dir).resolve()
    jpath = base / "metrics_by_day.json"
    if not jpath.is_file():
        print(f"Não encontrado: {jpath}", file=sys.stderr)
        return 1

    bundle = json.loads(jpath.read_text(encoding="utf-8"))
    rows = bundle.get("metrics_by_day") or []
    day = str(args.date).strip()
    row = next((r for r in rows if str(r.get("date", ""))[:10] == day[:10]), None)
    if not row:
        if args.demo:
            row = {
                "date": day,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred_threshold": 0.2,
                "pred_display_threshold": 0.2,
                "pred_display_mode": "demo",
            }
        else:
            print(f"Dia {day} não está em metrics_by_day.", file=sys.stderr)
            return 2

    if args.demo:
        real_pts = [(-3.72, -38.52), (-4.85, -39.27), (-5.9, -39.3)]
        pred_pts = [(-3.9, -38.6, 0.62), (-5.1, -39.05, 0.48), (-6.2, -39.5, 0.35)]
        row = dict(row)
        row.setdefault("iou", 0.0)
        row.setdefault("precision", 0.0)
        row.setdefault("recall", 0.0)
        row.setdefault("tp", 0)
        row.setdefault("fp", 0)
        row.setdefault("fn", 0)
        row.setdefault("pred_threshold", 0.2)
        row.setdefault("pred_display_threshold", 0.2)
        row.setdefault("pred_display_mode", "demo")
    else:
        rl = row.get("map_real_latlon")
        pl = row.get("map_pred_latlon_score")
        if not isinstance(rl, list) or not isinstance(pl, list):
            print(
                "JSON sem map_real_latlon / map_pred_latlon_score — volte a correr compare_st_hypernet_days "
                "ou use --demo para um PNG só com basemap de exemplo.",
                file=sys.stderr,
            )
            return 3

        real_pts = [(float(a[0]), float(a[1])) for a in rl if isinstance(a, (list, tuple)) and len(a) >= 2]
        pred_pts = [
            (float(t[0]), float(t[1]), float(t[2]) if len(t) > 2 else 0.5)
            for t in pl
            if isinstance(t, (list, tuple)) and len(t) >= 2
        ]

    from src.compare_ceara_maps import save_ceara_map_png

    demo_note = " (demonstração basemap)" if args.demo else ""
    title = f"Ceará — reais vs previsto (ST-HyperNet) — {day}{demo_note}  [fundo: ESRI World Imagery]"
    save_ceara_map_png(
        args.out,
        title,
        real_pts,
        pred_pts,
        metrics={k: v for k, v in row.items() if k not in ("map_real_latlon", "map_pred_latlon_score", "adaptive_threshold_meta")},
        pred_raster=None,
        display_thr=row.get("pred_display_threshold"),
        satellite_basemap=not args.no_satellite,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
