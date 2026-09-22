"""
Match a company name to its stock ticker.

Strategy (most reliable first):
  1. Download the official NSE master list (EQUITY_L.csv: SYMBOL + company name)
     and fuzzy-match the name against it -> "<SYMBOL>.NS".
  2. If no good match (e.g. a BSE-only or SME stock), fall back to Yahoo
     Finance's search API and prefer NSE (.NS) then BSE (.BO).

Public helpers:
    load_nse_master()            -> pandas.DataFrame [symbol, company, norm]
    match_ticker(name)           -> (ticker or None, score, source)
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(filename: str, fallback: str = None) -> str:
    candidates = [BASE_DIR / filename, Path.cwd() / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return fallback or str(BASE_DIR / filename)


NSE_CSV_FILE = _resolve_path("EQUITY_L.csv", r"C:\CODE\EQUITY_L.csv")
SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,application/json,*/*"}

# Score (0-100) below which we don't trust the NSE fuzzy match.
MATCH_THRESHOLD = 80

# Corporate suffixes / filler stripped before fuzzy matching.
_NOISE = re.compile(
    r"\b(limited|ltd|pvt|private|corporation|corp|company|co|the|and)\b",
    re.IGNORECASE,
)


def normalize(name: str) -> str:
    """Lowercase, drop punctuation and corporate filler, collapse spaces."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _download_nse_csv() -> None:
    req = urllib.request.Request(NSE_CSV_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(NSE_CSV_FILE, "wb") as fh:
        fh.write(data)


_MASTER_CACHE = None


def load_nse_master(refresh: bool = False) -> pd.DataFrame:
    """Return the NSE master list as a DataFrame with a normalized name column."""
    global _MASTER_CACHE
    if _MASTER_CACHE is not None and not refresh:
        return _MASTER_CACHE

    if refresh or not os.path.exists(NSE_CSV_FILE):
        print("Downloading NSE master symbol list...")
        _download_nse_csv()

    df = pd.read_csv(NSE_CSV_FILE)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"SYMBOL": "symbol", "NAME OF COMPANY": "company"})
    df = df[["symbol", "company"]].dropna()
    df["norm"] = df["company"].map(normalize)
    _MASTER_CACHE = df.reset_index(drop=True)
    return _MASTER_CACHE


def _yahoo_search(query: str) -> list:
    url = SEARCH_URL + "?" + urllib.parse.urlencode(
        {"q": query, "quotesCount": 8, "newsCount": 0})
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp).get("quotes", [])


def _yahoo_fallback(name: str):
    """Resolve via Yahoo search, preferring NSE then BSE. Returns ticker or None."""
    variants = []
    for v in (name, name.replace(" ", ""), name.replace(" ", "").upper(),
              name.split()[0] if name.split() else name):
        if v and v not in variants:
            variants.append(v)
    for q in variants:
        try:
            quotes = _yahoo_search(q)
        except Exception:
            quotes = []
        for exch in ("NSI", "BSE"):
            for quote in quotes:
                if quote.get("exchange") == exch and quote.get("symbol"):
                    return quote["symbol"]
        time.sleep(0.25)
    return None


def match_ticker(name: str):
    """
    Return (ticker, score, source) for a company name.
      source is "nse-symbol" (exact ticker-code hit), "nse" (name fuzzy match)
      or "yahoo" (search fallback). ticker is None if nothing matched.
    """
    df = load_nse_master()

    # 1. Exact ticker-code hit: "HG Infra" -> "HGINFRA" == symbol HGINFRA.
    #    100% reliable when the name collapses straight to a listed symbol.
    code = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    exact = df[df["symbol"].str.upper() == code]
    if not exact.empty:
        return f"{exact.iloc[0]['symbol']}.NS", 100.0, "nse-symbol"

    # 2. Yahoo search: its relevance ranking disambiguates similar names far
    #    better than fuzzy ("IRB Infrastructure" -> IRB, not MBLINFRA;
    #    "Adani Energy" -> ADANIENSOL, not ADANIGREEN). Covers BSE/SME too.
    yt = _yahoo_fallback(name)
    if yt:
        return yt, 100.0, "yahoo"

    # 3. Offline / Yahoo down -> fuzzy match the NSE master list.
    #    token_sort_ratio (not token_set) so a short query isn't handed a
    #    perfect score just for being a subset of a longer name.
    query = normalize(name)
    best = process.extractOne(
        query, df["norm"].tolist(), scorer=fuzz.token_sort_ratio)
    if best and best[1] >= MATCH_THRESHOLD:
        symbol = df.iloc[best[2]]["symbol"]
        return f"{symbol}.NS", round(best[1], 1), "nse-fuzzy"

    return None, (round(best[1], 1) if best else 0.0), "none"


if __name__ == "__main__":
    tests = ["NLC India", "IRB Infrastructure", "Rail Vikas Nigam", "JNK India",
             "Bharti Airtel", "Vodafone Idea", "NRB Bearings", "Creative Newtech",
             "HG Infra", "Merritronix", "EMS", "Alembic Pharma",
             "Adani Enterprises", "Adani Energy"]
    load_nse_master()
    for t in tests:
        ticker, score, source = match_ticker(t)
        print(f"{t:<22} -> {str(ticker):<16} ({source}, score {score})")
