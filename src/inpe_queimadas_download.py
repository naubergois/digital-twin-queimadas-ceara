"""
Download de focos de queimada do INPE (Programa Queimadas) para o Ceará.

Fonte principal (sem API REST privada — ficheiros HTTP públicos):

- **Anual por UF (satélite de referência):** ZIP com CSV já restrito ao estado, ex.::
    https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/EstadosBr_sat_ref/CE/focos_br_ce_ref_2024.zip

- **Mensal Brasil:** CSV nacional; filtra-se ``estado_id`` = Ceará (23)::
    https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/focos_mensal_br_YYYYMM.csv

O anual por UF costuma existir só até ao último ano fechado; anos mais recentes
usam agregação dos mensais.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.ceara_config import CEARA_UF_IBGE  # noqa: E402

BASE = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv"
ANNUAL_CE_ZIP = f"{BASE}/anual/EstadosBr_sat_ref/CE/focos_br_ce_ref_{{year}}.zip"
MONTHLY_BR_CSV = f"{BASE}/mensal/Brasil/focos_mensal_br_{{yyyymm}}.csv"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "digital-twin-queimadas-ceara/inpe_queimadas_download",
        }
    )
    return s


def download_bytes(url: str, session: Optional[requests.Session] = None) -> bytes:
    session = session or _session()
    r = session.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def annual_ce_zip_available(year: int, session: Optional[requests.Session] = None) -> bool:
    url = ANNUAL_CE_ZIP.format(year=year)
    session = session or _session()
    r = session.head(url, timeout=30, allow_redirects=True)
    return r.status_code == 200


def read_annual_ceara_dataframe(
    year: int,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Descarrega o ZIP anual do CE e devolve o conteúdo do CSV."""
    url = ANNUAL_CE_ZIP.format(year=year)
    raw = download_bytes(url, session=session)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"ZIP sem CSV: {url}")
        with zf.open(names[0]) as zfd:
            return pd.read_csv(zfd, low_memory=False)


def download_monthly_brasil_csv(
    yyyymm: str,
    dest_dir: Path,
    session: Optional[requests.Session] = None,
) -> Path:
    """Descarrega um CSV mensal nacional (sem filtrar estado)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = MONTHLY_BR_CSV.format(yyyymm=yyyymm)
    path = dest_dir / f"focos_mensal_br_{yyyymm}.csv"
    if path.is_file():
        return path.resolve()
    data = download_bytes(url, session=session)
    path.write_bytes(data)
    return path.resolve()


def filter_ceara_from_brasil_csv(
    csv_path: Path,
    uf_code: int = CEARA_UF_IBGE,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Lê CSV Brasil em blocos e mantém apenas linhas do Ceará."""
    parts: List[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if "estado_id" not in chunk.columns:
            raise ValueError(f"Coluna estado_id em falta em {csv_path}")
        eid = pd.to_numeric(chunk["estado_id"], errors="coerce")
        sub = chunk.loc[eid == float(uf_code)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def year_from_monthly_csvs(
    year: int,
    scratch_dir: Path,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Agrega os 12 meses (ou até falhar 404) num único DataFrame do CE."""
    session = session or _session()
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for month in range(1, 13):
        yyyymm = f"{year}{month:02d}"
        url = MONTHLY_BR_CSV.format(yyyymm=yyyymm)
        try:
            path = download_monthly_brasil_csv(yyyymm, scratch_dir, session=session)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                continue
            raise
        df = filter_ceara_from_brasil_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"Sem dados mensais INPE para Ceará em {year} (verifique se os ficheiros existem)."
        )
    return pd.concat(frames, ignore_index=True)


def download_ceara_years(
    start_year: int,
    end_year: int,
    out_dir: Optional[Path] = None,
    *,
    combine: bool = True,
    keep_monthly_files: bool = False,
) -> List[Path]:
    """
    Para cada ano entre ``start_year`` e ``end_year`` (inclusive):

    - Se existir ZIP anual do CE, usa esse ficheiro.
    - Caso contrário, junta os CSV mensais ``Brasil`` filtrados ao Ceará.

    Grava ``focos_ce_INPE_{ano}.csv`` em ``out_dir`` (por defeito
    ``data/inpe_focos_ce/anos/``). Se ``combine``, grava também
    ``focos_ce_INPE_{start_year}_{end_year}.csv``.
    """
    if start_year > end_year:
        raise ValueError("start_year não pode ser maior que end_year.")

    out_dir = Path(out_dir) if out_dir else (_REPO_ROOT / "data" / "inpe_focos_ce" / "anos")
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_scratch = out_dir.parent / "scratch_mensal_brasil"
    if not keep_monthly_files:
        monthly_scratch.mkdir(parents=True, exist_ok=True)

    session = _session()
    written: List[Path] = []
    combined: List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        out_year = out_dir / f"focos_ce_INPE_{year}.csv"
        if annual_ce_zip_available(year, session=session):
            df = read_annual_ceara_dataframe(year, session=session)
            df.to_csv(out_year, index=False)
        else:
            df = year_from_monthly_csvs(year, monthly_scratch, session=session)
            df.to_csv(out_year, index=False)

        written.append(out_year.resolve())
        combined.append(df)

        if not keep_monthly_files:
            for p in monthly_scratch.glob(f"focos_mensal_br_{year}*.csv"):
                p.unlink(missing_ok=True)

    if combine:
        all_df = pd.concat(combined, ignore_index=True)
        combo = out_dir.parent / f"focos_ce_INPE_{start_year}_{end_year}.csv"
        all_df.to_csv(combo, index=False)
        written.append(combo.resolve())

    return written


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Download focos INPE — Ceará (anual + mensal)")
    p.add_argument("--start", type=int, default=2024, help="Ano inicial (inclusive)")
    p.add_argument("--end", type=int, default=2026, help="Ano final (inclusive)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta para focos_ce_INPE_{ano}.csv (padrão: data/inpe_focos_ce/anos)",
    )
    p.add_argument(
        "--no-combine",
        action="store_true",
        help="Não gerar CSV único agregado no intervalo",
    )
    p.add_argument(
        "--keep-monthly",
        action="store_true",
        help="Manter CSVs mensais Brasil em disco (pasta scratch)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    paths = download_ceara_years(
        args.start,
        args.end,
        out_dir=args.out_dir,
        combine=not args.no_combine,
        keep_monthly_files=args.keep_monthly,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
