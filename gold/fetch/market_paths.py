"""Single on-disk tree for Yahoo gold, Vantage gold, and Vantage silver."""

from __future__ import annotations

from pathlib import Path

GOLD_ROOT = Path(__file__).resolve().parents[1]
MARKET_DATA = GOLD_ROOT / "data"
YAHOO_FINANCE_GOLD = MARKET_DATA / "Yahoo_Finance_Gold"
TRADINGVIEW_VANTAGE_GOLD = MARKET_DATA / "TradingView_Vantage_Gold"
TRADINGVIEW_VANTAGE_SILVER = MARKET_DATA / "TradingView_Vantage_Silver"

README_TEXT = """gold/data
========
One folder for every local metal feed. Do not mix files between subfolders.

gold/data/Yahoo_Finance_Gold
  Source: Yahoo Finance
  Symbol: GC=F
  What:   COMEX gold futures

gold/data/TradingView_Vantage_Gold
  Source: TradingView Vantage
  Symbol: VANTAGE:XAUUSD
  What:   Spot gold vs USD

gold/data/TradingView_Vantage_Silver
  Source: TradingView Vantage
  Symbol: VANTAGE:XAGUSD
  What:   Spot silver vs USD

The 6-hour hidden job updates all three.
"""


def write_readme() -> None:
    MARKET_DATA.mkdir(parents=True, exist_ok=True)
    (MARKET_DATA / "README.txt").write_text(README_TEXT, encoding="utf-8")

