"""
Treino e validação DTEC multi-dia com leave-one-day-out (DTEC §6).

Selecciona dias com NetCDFs presentes em ``data/goes16_raw/`` (após
``scripts/download_goes_multiday.py``) e corre LODO + relatório.

Comporta-se bem com 1 só dia: degrada para "treino in-sample" e avisa.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import List

import numpy as np

from src.inpe_dates import (
    load_inpe_with_dates,
    range_dense,
    stratified_by_month,
    top_active_days,
)
from src.multi_day_training import (
    MultiDayConfig,
    leave_one_day_out,
    load_day_data,
)
from src.unsupervised_fire_goes import find_local_goes_nc


REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _has_all_files(raw_dir: Path, day: date, hours, channels) -> bool:
    for h in hours:
        for ch in channels:
            try:
                find_local_goes_nc(raw_dir, day, ch, hour=int(h))
            except FileNotFoundError:
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="DTEC multi-dia leave-one-day-out.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--top", type=int, default=None)
    src.add_argument("--per-month", type=int, default=None)
    src.add_argument("--start", type=_parse_date, default=None)
    src.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--end", type=_parse_date, default=None)
    ap.add_argument("--months", nargs="+", type=int, default=None)
    ap.add_argument("--min-focos", type=int, default=10)
    ap.add_argument("--radius-km", type=float, default=10.0)
    ap.add_argument("--inpe-csv", type=Path,
                    default=REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "goes16_raw")
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "data" / "goes16_eval" / "multi_day_lodo.json")
    args = ap.parse_args(argv)

    df = load_inpe_with_dates(args.inpe_csv)

    if args.top is not None:
        candidate = top_active_days(df, n=args.top, min_focos=args.min_focos)
    elif args.per_month is not None:
        candidate = stratified_by_month(df, n_per_month=args.per_month,
                                        min_focos=args.min_focos, months=args.months)
    elif args.start is not None:
        if args.end is None:
            ap.error("--start exige --end")
        candidate = range_dense(df, args.start, args.end, min_focos=args.min_focos)
    elif args.days is not None:
        candidate = [_parse_date(s) for s in args.days]
    else:
        # Default: descobrir do que já existe localmente
        candidate = sorted({
            d for d in df["day"].unique()
            if _has_all_files(args.raw_dir, d, (16, 17, 18), (7, 13, 14))
        })

    # Filtra apenas dias com NetCDFs já presentes
    cfg = MultiDayConfig()
    days_with_data = [
        d for d in candidate
        if _has_all_files(args.raw_dir, d, cfg.hours, cfg.channels)
    ]
    missing = [d for d in candidate if d not in days_with_data]

    print(f"Candidatos: {len(candidate)} | com NetCDFs locais: {len(days_with_data)}")
    if missing:
        print(f"⚠️ {len(missing)} dias sem NetCDFs (faltam ficheiros):")
        for d in missing[:5]:
            print(f"   - {d}")
        if len(missing) > 5:
            print(f"   ... (+{len(missing)-5})")
        print(f"   Sugestão: python -m scripts.download_goes_multiday --days "
              + " ".join(d.isoformat() for d in missing))

    if not days_with_data:
        print("Nada para treinar. Saí.")
        return

    days_data = []
    for d in days_with_data:
        dd = load_day_data(d, df, args.raw_dir, cfg)
        if dd is not None:
            days_data.append(dd)
            print(f"   carregado {d.isoformat()}  focos={len(dd.df_day)} valid_cells={int(dd.valid_bins.sum())}")

    if len(days_data) < 2:
        print("\n⚠️ Apenas 1 dia carregado — LODO degenera. Reporta apenas treino in-sample.")
        from src.multi_day_training import train_multiday
        model = train_multiday(days_data, cfg=cfg)
        dd = days_data[0]
        peaks = model.predict_nms(dd.feats, dd.valid_bins,
                                  threshold=0.7258, nms_radius=1, smooth_sigma=1.2)
        from scipy.ndimage import binary_dilation, generate_binary_structure
        struct = generate_binary_structure(2, 2)
        pred = peaks.copy()
        for _ in range(2):
            pred = binary_dilation(pred, structure=struct)
        pred &= dd.valid_bins
        from src.event_centric import evaluate_event_centric
        from config.ceara_config import CEARA_BBOX
        m = evaluate_event_centric(pred, dd.df_day, CEARA_BBOX, cfg.grid_hw,
                                   radius_km=args.radius_km, valid_bins=dd.valid_bins)
        print(f"   in-sample R={args.radius_km}km: F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f}")
        return

    print(f"\n=== Leave-one-day-out — {len(days_data)} dias (R={args.radius_km} km) ===")
    rows = leave_one_day_out(days_data, cfg=cfg, radius_km=args.radius_km)
    print(f"\n{'dia':<12} {'#focos':>7} {'#pred':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for r in rows:
        print(f"{r['day']:<12} {r['n_focos']:>7d} {r['n_pred']:>6d} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

    f1s = np.array([r["f1"] for r in rows if r["n_focos"] > 0])
    ps = np.array([r["precision"] for r in rows if r["n_focos"] > 0])
    rs = np.array([r["recall"] for r in rows if r["n_focos"] > 0])
    print(f"\nLODO (n={len(f1s)} dias com focos)")
    print(f"  F1     médio={f1s.mean():.3f} ± {f1s.std():.3f}  mediano={float(np.median(f1s)):.3f}")
    print(f"  Prec   médio={ps.mean():.3f}")
    print(f"  Recall médio={rs.mean():.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "n_days": len(days_data),
        "radius_km": float(args.radius_km),
        "per_day": rows,
        "summary": {
            "f1_mean": float(f1s.mean()) if f1s.size else 0.0,
            "f1_std": float(f1s.std()) if f1s.size else 0.0,
            "f1_median": float(np.median(f1s)) if f1s.size else 0.0,
            "precision_mean": float(ps.mean()) if ps.size else 0.0,
            "recall_mean": float(rs.mean()) if rs.size else 0.0,
        },
    }, indent=2))
    print(f"\nGravado: {args.out_json}")


if __name__ == "__main__":
    main()
