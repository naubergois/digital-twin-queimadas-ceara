import pandas as pd

from src.compare_goes_unsupervised_days import fire_days_in_cube_span, _calendar_date


def test_fire_days_in_cube_span_filters_outside_window():
    counts = pd.Series({"2024-11-01": 100, "2024-12-28": 50, "2024-12-30": 80})
    times = pd.date_range("2024-12-20", "2024-12-31", freq="h")
    days = fire_days_in_cube_span(counts, times, max_days=5)
    assert all(pd.Timestamp(times[0]).date() <= _calendar_date(d) <= pd.Timestamp(times[-1]).date() for d in days)
    assert not any(_calendar_date(d) == _calendar_date("2024-11-01") for d in days)
    assert len(days) >= 1


def test_fire_days_in_cube_span_max_days_zero_returns_all_in_span():
    counts = pd.Series({"2024-12-28": 50, "2024-12-30": 80})
    times = pd.date_range("2024-12-20", "2024-12-31", freq="h")
    days = fire_days_in_cube_span(counts, times, max_days=0)
    assert len(days) == 2
    assert _calendar_date(days[0]) <= _calendar_date(days[1])
