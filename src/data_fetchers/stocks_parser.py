"""
Parse C:\\CODE\\Stocks.txt into a dict keyed by date, auto-resolving each
company name to a ticker from Yahoo Finance:

    {
        "09 June": {"NLC India": "NLCINDIA.NS", "IRB Infrastructure": "IRB.NS", ...},
        "08 June": {"Creative Newtech": "CNL.NS", ...},
    }

How the name -> ticker mapping works:
  1. Look the name up in ticker_cache.json (built incrementally).
  2. If missing, query Yahoo Finance's search API with several name
     variants ("HG Infra", "HGInfra", "HGINFRA", first word, ...),
     prefer an NSE listing (.NS), fall back to BSE (.BO), then any match.
  3. Save the result back to the cache so we never re-query it.

No tickers are hardcoded -- they are discovered from the net.
"""

import json
import os
import re
from pathlib import Path

from .ticker_matcher import match_ticker

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(filename: str, fallback: str = None) -> str:
    candidates = [BASE_DIR / filename, Path.cwd() / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return fallback or str(BASE_DIR / filename)


STOCKS_FILE = _resolve_path("Stocks.txt", r"C:\CODE\Stocks.txt")
CACHE_FILE = _resolve_path("ticker_cache.json", r"C:\CODE\ticker_cache.json")

# Matches: "Stocks to Watch Today: <names> in focus on <date>"
LINE_RE = re.compile(
    r"Stocks to Watch Today:\s*(?P<names>.+?)\s+in focus on\s+(?P<date>.+?)\s*$",
    re.IGNORECASE,
)


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8-sig") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            pass
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


def parse_stocks_file(path: str = STOCKS_FILE) -> dict:
    """Return {date: {stock_name: ticker}} parsed from the watch-list file."""
    cache = _load_cache()
    result = {}
    unresolved = []
    cache_dirty = False

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue

            date = m.group("date").strip()
            names = [n.strip() for n in m.group("names").split(",") if n.strip()]

            day_map = result.setdefault(date, {})
            for name in names:
                ticker = cache.get(name)
                if not ticker:
                    print(f"  matching '{name}'...")
                    ticker, score, source = match_ticker(name)
                    print(f"    -> {ticker}  ({source}, score {score})")
                    cache[name] = ticker
                    cache_dirty = True
                if not ticker:
                    unresolved.append(name)
                day_map[name] = ticker

    if cache_dirty:
        _save_cache(cache)
    if unresolved:
        print("Warning: could not resolve -> " + ", ".join(sorted(set(unresolved))))

    return result


if __name__ == "__main__":
    data = parse_stocks_file()
    print(json.dumps(data, indent=2, ensure_ascii=False))
