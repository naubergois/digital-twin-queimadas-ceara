"""
Utilitários para selecção de dias de treino DTEC multi-temporal.

A base INPE 2024–2026 tem ~111 mil focos no Ceará (198 dias activos só em
2024). O DTEC actual só vê 1 dia de GOES local (2024-10-31). Para subir o
F1 é preciso treinar e validar em **muitos dias** — este módulo expõe
estratégias deterministas de selecção:

- ``top_active_days``: dias com mais focos (cenas ricas em sinal positivo).
- ``stratified_by_month``: amostra equilibrada por mês (cobre sazonalidade).
- ``range_dense``: todos os dias num intervalo com pelo menos K focos.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def load_inpe_with_dates(path: Path) -> pd.DataFrame:
    """Carrega CSV INPE e adiciona coluna ``day`` (date) e ``datetime`` UTC-aware."""
    df = pd.read_csv(path, low_memory=False)
    raw = df["data_pas"] if "data_pas" in df.columns else df.get("data_hora_gmt")
    dt = pd.to_datetime(raw, utc=True, errors="coerce")
    df["datetime"] = dt
    df = df.dropna(subset=["datetime", "lat", "lon"])
    df["day"] = df["datetime"].dt.date
    return df.reset_index(drop=True)


def daily_focus_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("day").size().sort_values(ascending=False)


def top_active_days(df: pd.DataFrame, n: int = 30, *, min_focos: int = 10) -> List[date]:
    """Top-N dias por nº de focos, filtrando ruído (< ``min_focos``)."""
    s = daily_focus_counts(df)
    s = s[s >= int(min_focos)]
    return [d for d in s.head(int(n)).index.tolist()]


def stratified_by_month(
    df: pd.DataFrame,
    n_per_month: int = 4,
    *,
    min_focos: int = 10,
    months: Optional[Sequence[int]] = None,
) -> List[date]:
    """
    Para cada mês activo, devolve ``n_per_month`` dias com mais focos.
    Garante cobertura sazonal mesmo quando a base tem ruido em alguns meses.
    """
    s = daily_focus_counts(df)
    s = s[s >= int(min_focos)]
    by_month: dict = {}
    for d, n in s.items():
        m = d.month
        if months is not None and m not in months:
            continue
        by_month.setdefault(m, []).append(d)
    out: List[date] = []
    for m in sorted(by_month):
        out.extend(by_month[m][: int(n_per_month)])
    return sorted(out)


def range_dense(
    df: pd.DataFrame,
    start: date,
    end: date,
    *,
    min_focos: int = 10,
) -> List[date]:
    """Todos os dias entre ``[start, end]`` com pelo menos ``min_focos`` focos."""
    s = daily_focus_counts(df)
    s = s[(s.index >= start) & (s.index <= end) & (s >= int(min_focos))]
    return sorted(s.index.tolist())


def split_temporal_blocks(
    days: Sequence[date],
    n_folds: int = 4,
    *,
    buffer_days: int = 1,
) -> List[tuple]:
    """
    Divide ``days`` em ``n_folds`` blocos temporais contíguos. Cada fold
    devolve (treino, teste) e introduz ``buffer_days`` de gap entre os
    dois para evitar fuga por autocorrelação temporal.
    """
    sorted_days = sorted(set(days))
    n = len(sorted_days)
    if n_folds < 2 or n < 2 * n_folds:
        # poucos dias → fallback: leave-one-out
        folds = []
        for i, d in enumerate(sorted_days):
            train = [x for j, x in enumerate(sorted_days) if abs((x - d).days) > buffer_days]
            test = [d]
            folds.append((train, test))
        return folds
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = []
    for k in range(n_folds):
        test_block = sorted_days[edges[k]:edges[k + 1]]
        first_test = test_block[0]
        last_test = test_block[-1]
        train = [
            x for x in sorted_days
            if (x - last_test).days > buffer_days
            or (first_test - x).days > buffer_days
        ]
        folds.append((train, test_block))
    return folds
