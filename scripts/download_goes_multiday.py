"""
Download multi-dia GOES-16 ABI (canais 7, 13, 14) para um conjunto de
datas escolhido a partir do CSV INPE.

Uso típico:

    python -m scripts.download_goes_multiday --top 20
    python -m scripts.download_goes_multiday --month 10 11 12 --per-month 5
    python -m scripts.download_goes_multiday --start 2024-10-15 --end 2024-11-15

Por defeito grava em ``data/goes16_raw/`` (já no ``.gitignore``).
Por dia descarrega ``hours = (16, 17, 18)`` UTC × ``channels = (7, 13, 14)``.

Cada NetCDF pesa ~50 MB; um set de 20 dias × 3 horas × 3 canais ≈ 9 GB.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Sequence

from src.goes16_download import GOES16DownloadConfig, download_cmipf_channel
from src.inpe_dates import (
    load_inpe_with_dates,
    range_dense,
    stratified_by_month,
    top_active_days,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
DEFAULT_DEST = REPO_ROOT / "data" / "goes16_raw"
DEFAULT_HOURS = (16, 17, 18)
DEFAULT_CHANNELS = (7, 13, 14)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _download_day(
    cfg: GOES16DownloadConfig,
    day: date,
    *,
    hours: Sequence[int],
    channels: Sequence[int],
    overwrite: bool,
    show_progress: bool,
) -> int:
    n_ok = 0
    for hour in hours:
        when = datetime(day.year, day.month, day.day, int(hour), tzinfo=timezone.utc)
        for ch in channels:
            try:
                p = download_cmipf_channel(
                    when, ch, cfg=cfg, dest_dir=cfg.cache_dir,
                    overwrite=overwrite, show_progress=show_progress,
                )
                if p.is_file():
                    n_ok += 1
            except Exception as e:  # pragma: no cover
                print(f"  ⚠️ {day} {hour:02d}Z C{ch:02d} falhou: {e}")
    return n_ok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Download multi-dia GOES-16 ABI selecionado pela base INPE."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--top", type=int, help="N dias com mais focos (top-N)")
    src.add_argument("--per-month", type=int, help="N dias mais activos por mês")
    src.add_argument("--start", type=_parse_date, help="Início do intervalo (com --end)")
    src.add_argument("--days", nargs="+", help="Lista explícita YYYY-MM-DD")

    ap.add_argument("--end", type=_parse_date, help="Fim do intervalo (com --start)")
    ap.add_argument("--months", nargs="+", type=int, help="Filtrar a estes meses")
    ap.add_argument("--min-focos", type=int, default=10, help="Ignorar dias com < N focos")
    ap.add_argument("--inpe-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--hours", nargs="+", type=int, default=list(DEFAULT_HOURS))
    ap.add_argument("--channels", nargs="+", type=int, default=list(DEFAULT_CHANNELS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Lista os dias mas não descarrega")

    args = ap.parse_args(argv)

    df = load_inpe_with_dates(args.inpe_csv)

    if args.top:
        days = top_active_days(df, n=args.top, min_focos=args.min_focos)
    elif args.per_month:
        days = stratified_by_month(
            df, n_per_month=args.per_month,
            min_focos=args.min_focos, months=args.months,
        )
    elif args.start:
        if args.end is None:
            ap.error("--start exige --end")
        days = range_dense(df, args.start, args.end, min_focos=args.min_focos)
    else:
        days = [_parse_date(s) for s in args.days]

    print(f"Dias selecionados: {len(days)}")
    for d in days:
        nf = int((df["day"] == d).sum())
        print(f"  {d.isoformat()}  ({nf} focos INPE)")

    if args.dry_run:
        print("\n--dry-run: nada a descarregar.")
        return

    cfg = GOES16DownloadConfig(cache_dir=args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for d in days:
        print(f"\n=== {d.isoformat()} ===")
        n_ok = _download_day(
            cfg, d, hours=args.hours, channels=args.channels,
            overwrite=args.overwrite, show_progress=not args.no_progress,
        )
        total += n_ok

    print(f"\n✅ {total} NetCDFs prontos em {args.dest} (já no .gitignore).")


if __name__ == "__main__":
    main()
