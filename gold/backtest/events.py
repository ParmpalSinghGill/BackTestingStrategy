"""EventFinder-style gold levels for the local replay chart.

Read-only copy of the EventFinder rules (pivots, NEAR 0.20%, TOUCH, PDH/PDL,
Today's High/Low after a 1% arm). Adds Hourly support/resistance from 1-minute
bars. EventFinder itself is never imported or modified.

Day study (Today H/L, PDH/PDL, hourly today+yesterday) uses the CoinDCX /
TradingView XAUUSDT cut: 05:30 IST through next 05:29 IST (UTC calendar day).
Do not bucket those by IST midnight. See gold_chart/XAUUSDT_DAILY_CUT_PROMPT.md.

Label math ignores Saturday and Sunday (IST) completely — weekend bars do not
form pivots and do not cancel weekday levels. The replay chart can still show
those candles.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DISPLAY_TZ = "Asia/Kolkata"
XAUUSDT_DAILY_NAME = "Gold_Daily_XAUUSDT_UTC.csv"
XAUUSDT_DAILY_PATH = Path(__file__).resolve().parents[1] / "data" / "Yahoo_Finance_Gold" / XAUUSDT_DAILY_NAME
TRIGGER_TOL = 0.0020
WATCH_EXIT_DIST = 0.0040
SESSION_ARM = 0.01
NEAR_RETRIGGER_SEC = 3600
PREV_DAY_HIERARCHY_TOL = 0.0040
PIVOT_LEFT = 3
PIVOT_RIGHT = 2
PIVOT_BIG_MULT = 2.0
PIVOT_LEFT_GAP = 0.01
PIVOT_AVG_WINDOW = 20

AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

_cache: dict[str, object] = {"signature": None, "payload": None}


def _resample(daily: pd.DataFrame, rules: tuple[str, ...]) -> pd.DataFrame:
    last_err: Exception | None = None
    for rule in rules:
        try:
            return daily.resample(rule).agg(AGG).dropna()
        except (ValueError, KeyError) as exc:
            last_err = exc
    if last_err:
        raise last_err
    return daily.iloc[0:0].copy()


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("W-FRI").agg(AGG).dropna()


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample(daily, ("ME", "M"))


def to_yearly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample(daily, ("YE", "A", "Y"))


def to_2yearly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample(daily, ("2YE", "2A", "2Y"))


def to_hourly(minute: pd.DataFrame) -> pd.DataFrame:
    if minute.empty:
        return minute.iloc[0:0].copy()
    return minute.resample("1h").agg(AGG).dropna()


def drop_weekend_bars(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Remove Saturday/Sunday IST candles so labels use weekday liquidity only.

    Weekend prints can exist in MARKET_DATA (spot/crypto). Treat them as missing
    for EventFinder-style pivots, PDH/PDL, hourly S/R, and NEAR/TOUCH.
    """
    if df is None or df.empty:
        return df
    work = df.copy()
    if time_col is None:
        if "Datetime" in work.columns:
            time_col = "Datetime"
        elif "Date" in work.columns:
            time_col = "Date"

    if time_col and time_col in work.columns:
        ts = pd.to_datetime(work[time_col], utc=(time_col == "Datetime"), errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            weekday = ts.dt.tz_convert(DISPLAY_TZ).dt.dayofweek
        else:
            weekday = ts.dt.dayofweek
        return work.loc[weekday < 5].copy()

    idx = pd.to_datetime(work.index)
    if getattr(idx, "tz", None) is not None:
        weekday = idx.tz_convert(DISPLAY_TZ).dayofweek
    else:
        weekday = idx.dayofweek
    return work.loc[weekday < 5].copy()


def _align_ts(ts: pd.Timestamp, like: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    like = pd.Timestamp(like)
    if like.tzinfo is not None:
        if stamp.tzinfo is None:
            return stamp.tz_localize(like.tzinfo)
        return stamp.tz_convert(like.tzinfo)
    if stamp.tzinfo is not None:
        return stamp.tz_convert(DISPLAY_TZ).tz_localize(None)
    return stamp


def _to_ist(ts) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize(DISPLAY_TZ)
    return stamp.tz_convert(DISPLAY_TZ)


def utc_session_stamp(ts) -> pd.Timestamp:
    """UTC calendar date of a bar (naive midnight). 05:30 IST belongs to that UTC date."""
    return _to_ist(ts).tz_convert("UTC").normalize().tz_localize(None)


def utc_session_open(session_date, like) -> pd.Timestamp:
    """First minute of a UTC session: 00:00 UTC = 05:30 IST on that date."""
    start = pd.Timestamp(session_date)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    start = start.normalize()
    return _align_ts(start, like)


def utc_session_daily(minute_ix: pd.DataFrame) -> pd.DataFrame:
    """1m → XAUUSDT day candles: first Open, max High, min Low, last Close."""
    if minute_ix is None or minute_ix.empty:
        return minute_ix.iloc[0:0].copy() if minute_ix is not None else pd.DataFrame()
    work = minute_ix.copy()
    sessions = pd.DatetimeIndex([utc_session_stamp(ts) for ts in work.index])
    grouped = work.groupby(sessions).agg(AGG).dropna()
    grouped.index = pd.DatetimeIndex(grouped.index).tz_localize(None).normalize()
    grouped.index.name = "Date"
    return grouped


def write_xauusdt_utc_daily(
    minute: pd.DataFrame,
    path: Path | None = None,
) -> pd.DataFrame:
    """Write MARKET_DATA/.../Gold_Daily_XAUUSDT_UTC.csv from local 1m files."""
    daily = utc_session_daily(_indexed_minute(minute) if minute is not None else pd.DataFrame())
    target = Path(path) if path is not None else XAUUSDT_DAILY_PATH
    if daily.empty:
        return daily
    out = daily.reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return daily


def _hourly_recent_frame(hourly: pd.DataFrame, asof) -> pd.DataFrame:
    """Hourly bars from the previous UTC session open through `asof` (exclusive).

    Previous UTC open is yesterday 05:30 IST. Older hourly pivots are not valid.
    """
    if hourly.empty:
        return hourly.iloc[0:0].copy()
    like = asof if asof is not None else hourly.index[0]
    session = utc_session_stamp(asof)
    start = utc_session_open(session - pd.Timedelta(days=1), like)
    end = _align_ts(asof, like)
    return hourly.loc[(hourly.index >= start) & (hourly.index < end)]


def find_labels(
    df: pd.DataFrame,
    left: int = PIVOT_LEFT,
    right: int = PIVOT_RIGHT,
    big_mult: float = PIVOT_BIG_MULT,
    break_tol: float = 0.0,
    avg_window: int = PIVOT_AVG_WINDOW,
    left_gap: float = PIVOT_LEFT_GAP,
) -> list[dict[str, Any]]:
    n = len(df)
    if n < 5:
        return []
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    body = (df["Close"] - df["Open"]).abs().to_numpy(float)
    avg_body = (
        pd.Series(body).rolling(avg_window, min_periods=3).mean().bfill().to_numpy(float)
    )
    dates = df.index
    gap = float(left_gap or 0.0)

    def consec(arr: np.ndarray, i: int, step: int, cmp_lower: bool) -> int:
        count, j = 0, i + step
        while 0 <= j < n:
            ok = arr[j] < arr[i] if cmp_lower else arr[j] > arr[i]
            if not ok:
                break
            count += 1
            j += step
        return count

    def big(i: int, step: int, arr: np.ndarray, cmp_lower: bool) -> bool:
        j = i + step
        if not (0 <= j < n):
            return False
        beyond = arr[j] < arr[i] if cmp_lower else arr[j] > arr[i]
        return beyond and body[j] >= big_mult * avg_body[j]

    def gap_ok(i: int, arr: np.ndarray, after_lower: bool) -> bool:
        if gap <= 0 or i < 1 or i + 1 >= n:
            return False
        if arr[i] <= 0:
            return False
        after = arr[i + 1] < arr[i] if after_lower else arr[i + 1] > arr[i]
        if not after:
            return False
        return abs(arr[i - 1] - arr[i]) / arr[i] > gap

    def make(i: int, price: float, kind: str) -> dict[str, Any]:
        return {
            "type": kind,
            "price": float(price),
            "idx": i,
            "formed_date": dates[i],
            "canceled": False,
            "cancel_idx": None,
            "cancel_date": None,
        }

    labels: list[dict[str, Any]] = []
    for i in range(n):
        lc = consec(highs, i, -1, True)
        rc = consec(highs, i, +1, True)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big(i, +1, highs, True)) or (
            rc >= right and big(i, -1, highs, True)
        )
        if normal or big_ok or gap_ok(i, highs, after_lower=True):
            labels.append(make(i, highs[i], "resistance"))

        lc = consec(lows, i, -1, False)
        rc = consec(lows, i, +1, False)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big(i, +1, lows, False)) or (
            rc >= right and big(i, -1, lows, False)
        )
        if normal or big_ok or gap_ok(i, lows, after_lower=False):
            labels.append(make(i, lows[i], "support"))

    for lab in labels:
        i, price = lab["idx"], lab["price"]
        if lab["type"] == "resistance":
            beyond = highs[i + 1 :] > price * (1 + break_tol)
        else:
            beyond = lows[i + 1 :] < price * (1 - break_tol)
        hit = int(np.argmax(beyond)) if beyond.any() else -1
        if hit >= 0:
            k = i + 1 + hit
            lab["canceled"] = True
            lab["cancel_idx"] = k
            lab["cancel_date"] = dates[k]
    return labels


def _known_labels(labels: list[dict[str, Any]], frame: pd.DataFrame, asof) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    asof = _align_ts(asof, frame.index[0])
    out = []
    for lab in labels:
        confirm = lab["idx"] + PIVOT_RIGHT
        if confirm >= len(frame):
            continue
        if frame.index[confirm] >= asof:
            continue
        cancel = lab.get("cancel_date")
        if lab.get("canceled") and cancel is not None and _align_ts(cancel, asof) < asof:
            continue
        out.append(lab)
    return out


def nearest_levels(labels: list[dict[str, Any]], df: pd.DataFrame, ref: float):
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    dates = df.index

    def make(i: int, price: float, kind: str) -> dict[str, Any]:
        return {
            "type": kind,
            "price": float(price),
            "idx": i,
            "formed_date": dates[i],
            "canceled": False,
            "cancel_idx": None,
            "cancel_date": None,
        }

    def pick(kind: str, above: bool):
        cands = [
            lab
            for lab in labels
            if lab["type"] == kind
            and (lab["price"] > ref if above else lab["price"] < ref)
        ]
        if cands:
            return min(cands, key=lambda lab: lab["price"]) if above else max(
                cands, key=lambda lab: lab["price"]
            )
        if df.empty:
            return None
        if above:
            mask = highs > ref
            if not mask.any():
                return None
            j = int(np.argmax(np.where(mask, highs, -np.inf)))
            return make(j, highs[j], "resistance")
        mask = lows < ref
        if not mask.any():
            return None
        j = int(np.argmin(np.where(mask, lows, np.inf)))
        return make(j, lows[j], "support")

    return pick("resistance", True), pick("support", False)


def _valid_px(px: object) -> bool:
    try:
        value = float(px)
    except (TypeError, ValueError):
        return False
    return value > 0 and value != float("inf")


def _same_px(a: object, b: object) -> bool:
    if a is None or b is None:
        return False
    try:
        return round(float(a), 2) == round(float(b), 2)
    except (TypeError, ValueError):
        return False


def update_session_arm(armed: dict, today_high, today_low, current_price) -> dict:
    if not isinstance(armed, dict):
        armed = {"high": None, "low": None}
    px = float(current_price) if _valid_px(current_price) else None
    high = float(today_high) if _valid_px(today_high) else None
    low = float(today_low) if _valid_px(today_low) else None
    if high is None or not _same_px(armed.get("high"), high):
        armed["high"] = None
    if low is None or not _same_px(armed.get("low"), low):
        armed["low"] = None
    if px is not None and high is not None:
        drop_now = (high - px) / high if px < high else 0.0
        if drop_now >= SESSION_ARM:
            armed["high"] = high
    if px is not None and low is not None:
        rally_now = (px - low) / low if px > low else 0.0
        if rally_now >= SESSION_ARM:
            armed["low"] = low
    return armed


def session_extreme_levels(today_high, today_low, armed: dict | None) -> list[dict[str, Any]]:
    levels = []
    armed = armed if isinstance(armed, dict) else {"high": None, "low": None}
    high = float(today_high) if _valid_px(today_high) else None
    low = float(today_low) if _valid_px(today_low) else None
    if high is not None and _same_px(armed.get("high"), high):
        levels.append({
            "id": f"TH_{high:.2f}",
            "timeframe": "Today",
            "name": "Today's High",
            "type": "resistance",
            "price": high,
        })
    if low is not None and _same_px(armed.get("low"), low):
        levels.append({
            "id": f"TL_{low:.2f}",
            "timeframe": "Today",
            "name": "Today's Low",
            "type": "support",
            "price": low,
        })
    return levels


DAILY_TIMEFRAMES = (
    ("2Y", "2-Year", to_2yearly),
    ("1Y", "1-Year", to_yearly),
    ("M", "Monthly", to_monthly),
    ("W", "Weekly", to_weekly),
    ("D", "Daily", lambda df: df),
)


def _build_tf_sets(daily: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, list]]:
    sets = []
    for key, label, resampler in DAILY_TIMEFRAMES:
        try:
            frame = resampler(daily)
        except Exception:
            continue
        if frame.empty or len(frame) < 3:
            continue
        sets.append((key, label, frame, find_labels(frame)))
    return sets


def _unix_from_index(ts: pd.Timestamp) -> int:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(DISPLAY_TZ)
    else:
        stamp = stamp.tz_convert(DISPLAY_TZ)
    return int(stamp.tz_convert("UTC").timestamp())


def _sorted_sides(known: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    res = sorted({
        round(float(lab["price"]), 2)
        for lab in known
        if lab.get("type") == "resistance" and _valid_px(lab.get("price"))
    })
    sup = sorted({
        round(float(lab["price"]), 2)
        for lab in known
        if lab.get("type") == "support" and _valid_px(lab.get("price"))
    })
    return res, sup


def _nearest_price(sorted_prices: list[float], px: float, above: bool) -> float | None:
    if not sorted_prices:
        return None
    if above:
        index = bisect.bisect_right(sorted_prices, px)
        return sorted_prices[index] if index < len(sorted_prices) else None
    index = bisect.bisect_left(sorted_prices, px) - 1
    return sorted_prices[index] if index >= 0 else None


def _day_tf_state(tf_sets: list, asof) -> list[tuple[str, str, list[float], list[float]]]:
    ready = []
    for key, label, frame, labels in tf_sets:
        cut = _align_ts(asof, frame.index[0]) if not frame.empty else asof
        hist = frame.loc[frame.index < cut]
        if hist.empty or len(hist) < 3:
            continue
        known = [
            lab for lab in _known_labels(labels, frame, cut)
            if _align_ts(lab["formed_date"], cut) < cut
        ]
        res, sup = _sorted_sides(known)
        if not res and not sup:
            continue
        ready.append((key, label, res, sup))
    return ready


def _levels_from_day_state(
    day_state: list,
    current_price: float,
    today_high: float,
    today_low: float,
    pdh: float | None,
    pdl: float | None,
    armed: dict | None,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    higher: list[dict[str, Any]] = []
    daily_levels: list[dict[str, Any]] = []
    px = float(current_price)

    for key, label, res_prices, sup_prices in day_state:
        res_price = _nearest_price(res_prices, px, True)
        if res_price is not None and (not _valid_px(today_high) or today_high < res_price):
            item = {
                "id": f"{key}_R_{res_price:.2f}",
                "timeframe": label,
                "name": f"{label} Resistance",
                "type": "resistance",
                "price": res_price,
            }
            (higher if key in {"2Y", "1Y", "M", "W"} else daily_levels).append(item)
        sup_price = _nearest_price(sup_prices, px, False)
        if sup_price is not None and (not _valid_px(today_low) or today_low > sup_price):
            item = {
                "id": f"{key}_S_{sup_price:.2f}",
                "timeframe": label,
                "name": f"{label} Support",
                "type": "support",
                "price": sup_price,
            }
            (higher if key in {"2Y", "1Y", "M", "W"} else daily_levels).append(item)

    prev_day = []
    if pdh is not None and (not _valid_px(today_high) or today_high < pdh):
        prev_day.append({
            "id": f"PDH_{pdh:.2f}",
            "timeframe": "PrevDay",
            "name": "Prev Day High (PDH)",
            "type": "resistance",
            "price": float(pdh),
        })
    if pdl is not None and (not _valid_px(today_low) or today_low > pdl):
        prev_day.append({
            "id": f"PDL_{pdl:.2f}",
            "timeframe": "PrevDay",
            "name": "Prev Day Low (PDL)",
            "type": "support",
            "price": float(pdl),
        })

    kept_pd = []
    for level in prev_day:
        price = level["price"]
        nearby = any(
            abs(htf["price"] - price) / price <= PREV_DAY_HIERARCHY_TOL
            for htf in higher
        )
        if not nearby:
            kept_pd.append(level)

    extras = extra or []
    return higher + daily_levels + kept_pd + extras + session_extreme_levels(
        today_high, today_low, armed
    )


def _levels_at(
    tf_sets: list,
    asof,
    current_price: float,
    today_high: float,
    today_low: float,
    pdh: float | None,
    pdl: float | None,
    armed: dict | None,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return _levels_from_day_state(
        _day_tf_state(tf_sets, asof),
        current_price, today_high, today_low, pdh, pdl, armed, extra,
    )


def _seconds_between(prev_str: str | None, now_str: str) -> float | None:
    if not prev_str:
        return None
    a = pd.to_datetime(prev_str, errors="coerce")
    b = pd.to_datetime(now_str, errors="coerce")
    if pd.isna(a) or pd.isna(b):
        return None
    return float((b - a).total_seconds())


def _near_due(near_triggered: bool, lvl_state: dict, now_str: str) -> bool:
    if not near_triggered:
        return True
    prev = lvl_state.get("near_time") or lvl_state.get("last_updated")
    sec = _seconds_between(prev, now_str)
    if sec is None:
        return False
    return sec >= NEAR_RETRIGGER_SEC


def _touched(level: dict, bar_high: float, bar_low: float) -> bool:
    price = float(level["price"])
    kind = (level.get("type") or "").lower()
    if kind == "resistance":
        return bar_high >= price
    if kind == "support":
        return bar_low <= price
    return bar_low <= price <= bar_high


def _scan_bar(
    levels: list[dict[str, Any]],
    state: dict,
    current_price: float,
    current_time_str: str,
    bar_high: float,
    bar_low: float,
    bar_time: int,
    events: list[dict[str, Any]],
) -> None:
    for level in levels:
        lid = level["id"]
        lprice = float(level["price"])
        samples = [current_price, bar_high, bar_low]
        near_dist = min(abs(sample - lprice) / lprice for sample in samples if lprice)
        close_dist = abs(current_price - lprice) / lprice if lprice else 1.0
        is_near = near_dist <= TRIGGER_TOL
        touched = _touched(level, bar_high, bar_low)

        lvl_state = state.get(lid, {
            "near_triggered": False,
            "touch_triggered": False,
        })
        near_triggered = bool(lvl_state.get("near_triggered"))
        touch_triggered = bool(lvl_state.get("touch_triggered"))

        def emit(status: str) -> None:
            events.append({
                "time": int(bar_time),
                "status": status,
                "level": level["name"],
                "timeframe": level["timeframe"],
                "type": level["type"],
                "price": round(lprice, 2),
                "spot": round(current_price, 2),
                "distPct": round(close_dist * 100, 3),
            })

        if touch_triggered:
            pass
        elif touched:
            if _near_due(near_triggered, lvl_state, current_time_str):
                emit("NEAR")
                near_triggered = True
                lvl_state["near_time"] = current_time_str
            emit("TOUCH")
            touch_triggered = True
        elif is_near:
            if _near_due(near_triggered, lvl_state, current_time_str):
                emit("NEAR")
                near_triggered = True
                lvl_state["near_time"] = current_time_str
        elif near_triggered and close_dist > WATCH_EXIT_DIST:
            near_triggered = False

        lvl_state["near_triggered"] = near_triggered
        lvl_state["touch_triggered"] = touch_triggered
        lvl_state["last_updated"] = current_time_str
        state[lid] = lvl_state


def _indexed_daily(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.tz_localize(None).dt.normalize()
    return (
        work.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        .dropna()
        .pipe(drop_weekend_bars)
    )


def _indexed_minute(minute: pd.DataFrame) -> pd.DataFrame:
    work = minute.copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    work["Datetime"] = work["Datetime"].dt.tz_convert(DISPLAY_TZ)
    return (
        work.sort_values("Datetime")
        .drop_duplicates(subset=["Datetime"], keep="last")
        .set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]
        .dropna()
    )


def _hourly_extra(hourly: pd.DataFrame, asof, current_price: float) -> list[dict[str, Any]]:
    window = _hourly_recent_frame(hourly, asof)
    if window.empty or len(window) < 5:
        return []
    known = _known_labels(find_labels(window), window, asof)
    res_prices, sup_prices = _sorted_sides(known)
    extra = []
    res_price = _nearest_price(res_prices, float(current_price), True)
    if res_price is not None:
        extra.append({
            "id": f"H_R_{res_price:.2f}",
            "timeframe": "Hourly",
            "name": "Hourly Resistance",
            "type": "resistance",
            "price": res_price,
        })
    sup_price = _nearest_price(sup_prices, float(current_price), False)
    if sup_price is not None:
        extra.append({
            "id": f"H_S_{sup_price:.2f}",
            "timeframe": "Hourly",
            "name": "Hourly Support",
            "type": "support",
            "price": sup_price,
        })
    return extra


def scan_events(minute: pd.DataFrame, daily: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    daily_ix = _indexed_daily(daily) if daily is not None and not daily.empty else pd.DataFrame()
    minute_ix = drop_weekend_bars(
        _indexed_minute(minute) if minute is not None and not minute.empty else pd.DataFrame()
    )
    if daily_ix.empty:
        return events

    tf_sets = _build_tf_sets(daily_ix)
    if minute_ix.empty:
        return events

    hourly = to_hourly(minute_ix)
    session_ix = drop_weekend_bars(utc_session_daily(minute_ix))
    last_day = None
    last_hour = None
    pdh = pdl = None
    armed = {"high": None, "low": None}
    today_high = float("-inf")
    today_low = float("inf")
    hour_extra: list[dict[str, Any]] = []
    hour_res: list[float] = []
    hour_sup: list[float] = []
    day_state: list = []
    state = {}

    for row in minute_ix.itertuples():
        ts = row.Index
        local = ts.tz_convert(DISPLAY_TZ) if getattr(ts, "tzinfo", None) else pd.Timestamp(ts)
        day = utc_session_stamp(local)
        hour = local.floor("h")
        px = float(row.Close)
        high = float(row.High)
        low = float(row.Low)

        if last_day != day:
            hist = session_ix.loc[session_ix.index < day] if not session_ix.empty else session_ix
            if hist.empty:
                pdh = None
                pdl = None
            else:
                prev = hist.iloc[-1]
                pdh = float(prev["High"])
                pdl = float(prev["Low"])
            armed = {"high": None, "low": None}
            today_high = float("-inf")
            today_low = float("inf")
            day_state = _day_tf_state(tf_sets, local)
            last_day = day
            last_hour = None

        if last_hour != hour:
            window = _hourly_recent_frame(hourly, hour)
            if len(window) >= 5:
                known = _known_labels(find_labels(window), window, hour)
                hour_res, hour_sup = _sorted_sides(known)
            else:
                hour_res, hour_sup = [], []
            last_hour = hour

        hour_extra = []
        hour_r = _nearest_price(hour_res, px, True)
        if hour_r is not None:
            hour_extra.append({
                "id": f"H_R_{hour_r:.2f}",
                "timeframe": "Hourly",
                "name": "Hourly Resistance",
                "type": "resistance",
                "price": hour_r,
            })
        hour_s = _nearest_price(hour_sup, px, False)
        if hour_s is not None:
            hour_extra.append({
                "id": f"H_S_{hour_s:.2f}",
                "timeframe": "Hourly",
                "name": "Hourly Support",
                "type": "support",
                "price": hour_s,
            })

        levels = _levels_from_day_state(
            day_state, px, today_high, today_low, pdh, pdl, armed, hour_extra,
        )
        time_str = local.strftime("%Y-%m-%d %H:%M:%S")
        bar_unix = int(ts.tz_convert("UTC").timestamp()) if getattr(ts, "tzinfo", None) else _unix_from_index(ts)
        _scan_bar(levels, state, px, time_str, high, low, bar_unix, events)
        today_high = high if today_high == float("-inf") else max(today_high, high)
        today_low = low if today_low == float("inf") else min(today_low, low)
        update_session_arm(armed, today_high, today_low, px)

    events.sort(key=lambda item: (item["time"], 0 if item["status"] == "NEAR" else 1))
    return events


def _serialize_pivot_labels(
    key: str,
    label: str,
    frame: pd.DataFrame,
    labels: list[dict[str, Any]],
    hourly: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    n = len(frame)
    for lab in labels:
        confirm = int(lab["idx"]) + PIVOT_RIGHT
        if confirm >= n:
            continue
        price = float(lab["price"])
        if not _valid_px(price):
            continue
        kind = str(lab.get("type") or "")
        if kind not in {"support", "resistance"}:
            continue
        formed = lab["formed_date"]
        cancel = lab.get("cancel_date") if lab.get("canceled") else None
        out.append({
            "id": f"{key}_{kind[0].upper()}_{price:.2f}_{_unix_from_index(formed)}",
            "timeframe": label,
            "name": f"{label} {'Resistance' if kind == 'resistance' else 'Support'}",
            "type": kind,
            "price": round(price, 2),
            "sourceTime": _unix_from_index(formed),
            "readyTime": _unix_from_index(frame.index[confirm]),
            "cancelTime": _unix_from_index(cancel) if cancel is not None else None,
            "hourly": hourly,
        })
    return out


def _unix_utc_session(session_date) -> int:
    start = pd.Timestamp(session_date)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    return int(start.normalize().timestamp())


def _prev_day_labels(daily_ix: pd.DataFrame, minute_ix: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if daily_ix is None or len(daily_ix) < 2:
        return out
    dates = list(daily_ix.index)
    meta: list[dict[str, Any]] = []
    for index in range(1, len(dates)):
        day = pd.Timestamp(dates[index]).tz_localize(None).normalize()
        prev_day = pd.Timestamp(dates[index - 1]).tz_localize(None).normalize()
        nxt = (
            pd.Timestamp(dates[index + 1]).tz_localize(None).normalize()
            if index + 1 < len(dates)
            else day + pd.Timedelta(days=1)
        )
        day_end = _unix_utc_session(nxt)
        day_high = float(daily_ix.iloc[index]["High"])
        day_low = float(daily_ix.iloc[index]["Low"])
        prev = daily_ix.iloc[index - 1]
        pdh = float(prev["High"])
        pdl = float(prev["Low"])
        meta.append({
            "day": day,
            "source": _unix_utc_session(prev_day),
            "ready": _unix_utc_session(day),
            "pdh": pdh,
            "pdl": pdl,
            "pdh_cancel": _unix_utc_session(day) if day_high >= pdh else day_end,
            "pdl_cancel": _unix_utc_session(day) if day_low <= pdl else day_end,
        })

    if minute_ix is not None and not minute_ix.empty:
        by_day = {item["day"]: item for item in meta}
        seen_high: set[pd.Timestamp] = set()
        seen_low: set[pd.Timestamp] = set()
        for row in minute_ix.itertuples():
            ts = row.Index
            local = ts.tz_convert(DISPLAY_TZ) if getattr(ts, "tzinfo", None) else pd.Timestamp(ts)
            day = utc_session_stamp(local)
            item = by_day.get(day)
            if item is None:
                continue
            unix = (
                int(ts.tz_convert("UTC").timestamp())
                if getattr(ts, "tzinfo", None)
                else _unix_from_index(ts)
            )
            if day not in seen_high and float(row.High) >= item["pdh"]:
                item["pdh_cancel"] = unix
                seen_high.add(day)
            if day not in seen_low and float(row.Low) <= item["pdl"]:
                item["pdl_cancel"] = unix
                seen_low.add(day)

    for item in meta:
        if _valid_px(item["pdh"]):
            out.append({
                "id": f"PDH_{item['pdh']:.2f}_{item['source']}",
                "timeframe": "PrevDay",
                "name": "Prev Day High (PDH)",
                "type": "resistance",
                "price": round(float(item["pdh"]), 2),
                "sourceTime": item["source"],
                "readyTime": item["ready"],
                "cancelTime": item["pdh_cancel"],
                "hourly": False,
            })
        if _valid_px(item["pdl"]):
            out.append({
                "id": f"PDL_{item['pdl']:.2f}_{item['source']}",
                "timeframe": "PrevDay",
                "name": "Prev Day Low (PDL)",
                "type": "support",
                "price": round(float(item["pdl"]), 2),
                "sourceTime": item["source"],
                "readyTime": item["ready"],
                "cancelTime": item["pdl_cancel"],
                "hourly": False,
            })
    return out


def collect_level_labels(minute: pd.DataFrame, daily: pd.DataFrame) -> list[dict[str, Any]]:
    """Intact EventFinder pivots with source/cancel times for replay H-rays."""
    daily_ix = _indexed_daily(daily) if daily is not None and not daily.empty else pd.DataFrame()
    minute_ix = drop_weekend_bars(
        _indexed_minute(minute) if minute is not None and not minute.empty else pd.DataFrame()
    )
    out: list[dict[str, Any]] = []
    if not daily_ix.empty:
        for key, label, frame, labels in _build_tf_sets(daily_ix):
            out.extend(_serialize_pivot_labels(key, label, frame, labels))
    if not minute_ix.empty:
        out.extend(_prev_day_labels(drop_weekend_bars(utc_session_daily(minute_ix)), minute_ix))
        hourly = to_hourly(minute_ix)
        if len(hourly) >= 5:
            out.extend(
                _serialize_pivot_labels("H", "Hourly", hourly, find_labels(hourly), hourly=True)
            )
    return out


def serialize_events(
    minute: pd.DataFrame,
    daily: pd.DataFrame,
    signature,
    utc_daily_path=None,
) -> dict[str, object]:
    cache_key = (signature, "utc-session-weekdays-v1")
    if _cache["signature"] == cache_key and _cache["payload"] is not None:
        return _cache["payload"]  # type: ignore[return-value]
    print("[gold-chart] Scanning EventFinder levels (2Y/1Y/M/W/D + hourly today/yesterday)…", flush=True)
    try:
        xau_daily = write_xauusdt_utc_daily(minute, utc_daily_path)
        print(f"[gold-chart] {XAUUSDT_DAILY_NAME}: {len(xau_daily)} UTC sessions", flush=True)
    except OSError as exc:
        print(f"[gold-chart] Could not write {XAUUSDT_DAILY_NAME}: {exc}", flush=True)
    events = scan_events(minute, daily)
    labels = collect_level_labels(minute, daily)
    print(f"[gold-chart] {len(events)} gold events · {len(labels)} S/R labels", flush=True)
    payload = {
        "triggerTol": TRIGGER_TOL,
        "watchExit": WATCH_EXIT_DIST,
        "count": len(events),
        "events": events,
        "labels": labels,
    }
    _cache.update({"signature": cache_key, "payload": payload})
    return payload
