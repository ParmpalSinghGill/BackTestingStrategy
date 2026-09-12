"""Mark-to-market helpers for 21-column account statements.

Holding_Equity_Value on date D is sum(qty * close_D) per Guide/Account_Statement_guide.md.
Sizing still uses cost basis so fills stay unchanged; only reported equity is MTM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DAILY_DIR = BASE_DIR / "data_daily"


class CloseCache:
    def __init__(self) -> None:
        self._bars: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def preload(self, tickers) -> None:
        for t in tickers:
            if t and t != "N/A":
                self._load(str(t))

    def _load(self, ticker: str) -> None:
        if ticker in self._bars:
            return
        candidates = [
            DATA_DAILY_DIR / f"{ticker}_1d.csv",
            DATA_DAILY_DIR / f"{ticker}.csv",
            DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}_1d.csv",
            DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            self._bars[ticker] = (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
            return
        df = pd.read_csv(path)
        dcol = "Date" if "Date" in df.columns else df.columns[0]
        ccol = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if ccol is None:
            self._bars[ticker] = (np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
            return
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
        df = df.dropna(subset=[dcol]).sort_values(dcol)
        self._bars[ticker] = (
            df[dcol].to_numpy(dtype="datetime64[ns]"),
            pd.to_numeric(df[ccol], errors="coerce").to_numpy(dtype=float),
        )

    def close_on(self, ticker: str, day, fallback: float) -> float:
        self._load(ticker)
        dates, closes = self._bars[ticker]
        if len(dates) == 0:
            return float(fallback)
        d = np.datetime64(pd.Timestamp(day).normalize(), "ns")
        idx = int(np.searchsorted(dates, d, side="right")) - 1
        if idx < 0:
            return float(fallback)
        px = closes[idx]
        if not np.isfinite(px) or px <= 0:
            return float(fallback)
        return float(px)

    def holding_mtm(self, open_pos: dict, day) -> float:
        tot = 0.0
        for p in open_pos.values():
            px = self.close_on(p["Ticker"], day, p["Entry_Price"])
            tot += float(p["Quantity"]) * px
        return round(tot, 2)


def expand_daily_mtm(
    snaps: list[dict],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    closes: CloseCache,
) -> pd.DataFrame:
    """Calendar daily equity: cash from last event, holdings marked to that day's close."""
    ordered = sorted(snaps, key=lambda s: s["Date"])
    cal = pd.date_range(start_dt.normalize(), end_dt.normalize(), freq="D")
    rows: list[dict] = []
    si = 0
    cash = float(ordered[0]["Cash_Balance"]) if ordered else 0.0
    opens: list[tuple] = list(ordered[0]["Open"]) if ordered else []
    n_open = int(ordered[0]["Active_Positions"]) if ordered else 0
    prev_bal = None
    for d in cal:
        while si < len(ordered) and pd.Timestamp(ordered[si]["Date"]).normalize() <= d:
            cash = float(ordered[si]["Cash_Balance"])
            opens = list(ordered[si]["Open"])
            n_open = int(ordered[si]["Active_Positions"])
            si += 1
        hold = round(sum(q * closes.close_on(t, d, ep) for t, q, ep in opens), 2)
        bal = round(cash + hold, 2)
        if prev_bal is None:
            pnl, ret = 0.0, 0.0
        else:
            pnl = round(bal - prev_bal, 2)
            ret = (pnl / prev_bal * 100.0) if prev_bal else 0.0
        rows.append({
            "Date": d,
            "Balance": bal,
            "Cash_Balance": cash,
            "Holding_Equity_Value": hold,
            "Active_Positions": n_open,
            "Daily_PnL": pnl,
            "Daily_Return_Pct": ret,
        })
        prev_bal = bal
    return pd.DataFrame(rows)


def index_trade_charts(chart_dir: Path) -> dict[tuple[str, str], str]:
    """Map (ticker_without_suffix, YYYY-MM-DD entry) -> filename."""
    out: dict[tuple[str, str], str] = {}
    if not chart_dir.exists():
        return out
    for p in chart_dir.glob("Trade_*.png"):
        parts = p.stem.split("_")
        if len(parts) < 5:
            continue
        out[(parts[2], parts[3])] = p.name
    return out


def max_drawdown_pct(balances) -> float:
    peak = None
    dd = 0.0
    for b in balances:
        b = float(b)
        peak = b if peak is None else max(peak, b)
        if peak and peak > 0:
            dd = max(dd, (peak - b) / peak * 100.0)
    return float(dd)
