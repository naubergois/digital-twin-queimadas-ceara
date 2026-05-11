"""Testes do pipeline multi-dia (selecção de datas + treino LODO)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inpe_dates import (
    daily_focus_counts,
    range_dense,
    split_temporal_blocks,
    stratified_by_month,
    top_active_days,
)


def _toy_focos_df() -> pd.DataFrame:
    rows = []
    # 3 focos em 2024-10-31, 1 em 2024-11-01, 50 em 2024-11-15, 20 em 2024-12-12
    for _ in range(3):
        rows.append({"data_pas": "2024-10-31 17:00:00", "lat": -5.0, "lon": -39.0})
    for _ in range(1):
        rows.append({"data_pas": "2024-11-01 17:00:00", "lat": -5.0, "lon": -39.0})
    for _ in range(50):
        rows.append({"data_pas": "2024-11-15 17:00:00", "lat": -5.0, "lon": -39.0})
    for _ in range(20):
        rows.append({"data_pas": "2024-12-12 17:00:00", "lat": -5.0, "lon": -39.0})
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["data_pas"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon"])
    df["day"] = df["datetime"].dt.date
    return df.reset_index(drop=True)


def test_daily_focus_counts_orders_by_count():
    df = _toy_focos_df()
    s = daily_focus_counts(df)
    assert list(s.index[:3]) == [date(2024, 11, 15), date(2024, 12, 12), date(2024, 10, 31)]
    assert int(s.iloc[0]) == 50


def test_top_active_days_filters_min_focos():
    df = _toy_focos_df()
    out = top_active_days(df, n=10, min_focos=10)
    # 1 foco no 2024-11-01 e 3 no 2024-10-31 < 10 → excluídos
    assert date(2024, 11, 15) in out
    assert date(2024, 12, 12) in out
    assert date(2024, 11, 1) not in out
    assert date(2024, 10, 31) not in out


def test_top_active_days_respects_n():
    df = _toy_focos_df()
    out = top_active_days(df, n=1, min_focos=1)
    assert len(out) == 1
    assert out[0] == date(2024, 11, 15)


def test_stratified_by_month_caps_per_month():
    df = _toy_focos_df()
    # 3 meses distintos (10, 11, 12); n_per_month=1 → 3 dias na saída
    out = stratified_by_month(df, n_per_month=1, min_focos=1)
    assert len(out) == 3
    months = {d.month for d in out}
    assert months == {10, 11, 12}
    # n_per_month=2 → mês 11 contribui 2 dias, outros 1 → 4 dias
    out2 = stratified_by_month(df, n_per_month=2, min_focos=1)
    assert len(out2) == 4


def test_stratified_filter_months():
    df = _toy_focos_df()
    out = stratified_by_month(df, n_per_month=10, min_focos=1, months=[11])
    assert all(d.month == 11 for d in out)
    assert len(out) == 2  # 2024-11-01 e 2024-11-15


def test_range_dense_window():
    df = _toy_focos_df()
    out = range_dense(df, date(2024, 11, 1), date(2024, 11, 30), min_focos=10)
    assert out == [date(2024, 11, 15)]


def test_split_temporal_blocks_disjoint_test_partition():
    days = [date(2024, 11, 1), date(2024, 11, 5), date(2024, 11, 10),
            date(2024, 11, 15), date(2024, 11, 20), date(2024, 11, 25),
            date(2024, 11, 30), date(2024, 12, 5)]
    folds = split_temporal_blocks(days, n_folds=4, buffer_days=1)
    assert len(folds) == 4
    # Cada test_block está nos dias da lista
    all_test = []
    for _, test in folds:
        for t in test:
            assert t in days
            all_test.append(t)
    # Os 4 test_blocks devem cobrir todos os dias (partição)
    assert sorted(all_test) == sorted(days)


def test_split_temporal_blocks_buffer_excludes_neighbors():
    days = [date(2024, 11, 1), date(2024, 11, 2), date(2024, 11, 3),
            date(2024, 11, 4), date(2024, 11, 5), date(2024, 11, 6),
            date(2024, 11, 7), date(2024, 11, 8)]
    folds = split_temporal_blocks(days, n_folds=4, buffer_days=2)
    for train, test in folds:
        for t in test:
            for tr in train:
                assert abs((tr - t).days) > 2, f"buffer falhou: {tr} vs {t}"


def test_split_temporal_blocks_lodo_fallback_when_few_days():
    days = [date(2024, 11, 1), date(2024, 11, 5)]
    folds = split_temporal_blocks(days, n_folds=4)
    # Fallback: LODO com buffer
    assert len(folds) == 2
    for train, test in folds:
        assert len(test) == 1
