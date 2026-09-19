import pandas as pd

from gold_chart.events import (
    drop_weekend_bars,
    utc_session_daily,
    utc_session_open,
    utc_session_stamp,
    _day_tf_state,
    _hourly_recent_frame,
    find_labels,
)


def test_session_date_uses_0530_ist_cut():
    before = pd.Timestamp("2026-09-19 05:29:00+05:30")
    open_bar = pd.Timestamp("2026-09-19 05:30:00+05:30")
    assert utc_session_stamp(before) == pd.Timestamp("2026-09-18")
    assert utc_session_stamp(open_bar) == pd.Timestamp("2026-09-19")


def test_session_daily_ohlc_from_1m():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-09-18 05:30:00+05:30"),
        pd.Timestamp("2026-09-18 12:00:00+05:30"),
        pd.Timestamp("2026-09-19 05:29:00+05:30"),
        pd.Timestamp("2026-09-19 05:30:00+05:30"),
    ])
    frame = pd.DataFrame({
        "Open": [100.0, 102.0, 104.0, 200.0],
        "High": [101.0, 110.0, 105.0, 201.0],
        "Low": [99.0, 101.0, 98.0, 199.0],
        "Close": [102.0, 103.0, 104.0, 200.5],
        "Volume": [1, 1, 1, 1],
    }, index=idx)
    daily = utc_session_daily(frame)
    monday = daily.loc[pd.Timestamp("2026-09-18")]
    assert monday["Open"] == 100.0
    assert monday["High"] == 110.0
    assert monday["Low"] == 98.0
    assert monday["Close"] == 104.0
    assert pd.Timestamp("2026-09-19") in daily.index


def test_hourly_window_starts_previous_utc_open():
    idx = pd.date_range("2026-09-17 05:30:00+05:30", periods=60, freq="h")
    hourly = pd.DataFrame({
        "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1,
    }, index=idx)
    asof = pd.Timestamp("2026-09-19 10:00:00+05:30")
    window = _hourly_recent_frame(hourly, asof)
    assert window.index.min() == pd.Timestamp("2026-09-18 05:30:00+05:30")
    assert window.index.max() < asof
    open_ist = utc_session_open(pd.Timestamp("2026-09-18"), asof)
    assert open_ist == pd.Timestamp("2026-09-18 05:30:00+05:30")


def test_day_tf_state_accepts_tz_aware_asof():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.Series(range(40), index=idx, dtype=float) + 100
    frame = pd.DataFrame({
        "Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1,
    })
    tf_sets = [("D", "Daily", frame, find_labels(frame))]
    asof = pd.Timestamp("2024-02-01 10:00:00+05:30")
    ready = _day_tf_state(tf_sets, asof)
    assert ready
    assert ready[0][0] == "D"


def test_drop_weekend_bars_removes_sat_sun_ist():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-09-18 10:00:00+05:30"),  # Friday
        pd.Timestamp("2026-09-19 10:00:00+05:30"),  # Saturday
        pd.Timestamp("2026-09-20 10:00:00+05:30"),  # Sunday
        pd.Timestamp("2026-09-21 10:00:00+05:30"),  # Monday
    ])
    frame = pd.DataFrame({
        "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.2, "Volume": 1,
    }, index=idx)
    out = drop_weekend_bars(frame)
    assert list(out.index) == [
        pd.Timestamp("2026-09-18 10:00:00+05:30"),
        pd.Timestamp("2026-09-21 10:00:00+05:30"),
    ]


def test_drop_weekend_bars_removes_sat_sun_daily():
    idx = pd.DatetimeIndex(["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"])
    frame = pd.DataFrame({
        "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.2, "Volume": 1,
    }, index=idx)
    out = drop_weekend_bars(frame)
    assert [d.date().isoformat() for d in out.index] == ["2026-09-18", "2026-09-21"]


def test_find_labels_skips_weekend_pivot_and_cancel():
    # Friday high, Saturday even higher (would cancel Friday R if weekend counted),
    # Monday lower — weekday-only series should keep Friday resistance.
    idx = pd.DatetimeIndex([
        "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
        "2026-09-19", "2026-09-21", "2026-09-22", "2026-09-23",
    ])
    high = [10, 11, 12, 11, 20, 30, 13, 12, 11]
    low = [9, 10, 11, 10, 15, 14, 12, 11, 10]
    close = [9.5, 10.5, 11.5, 10.5, 16, 20, 12.5, 11.5, 10.5]
    full = pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close, "Volume": 1,
    }, index=pd.DatetimeIndex(idx))
    weekdays = drop_weekend_bars(full)
    full_labels = find_labels(full)
    week_labels = find_labels(weekdays)
    weekend_highs = {round(lab["price"], 2) for lab in full_labels if lab["type"] == "resistance"}
    weekday_highs = {round(lab["price"], 2) for lab in week_labels if lab["type"] == "resistance"}
    assert 30.0 in weekend_highs or any(lab.get("canceled") for lab in full_labels)
    assert 30.0 not in weekday_highs

