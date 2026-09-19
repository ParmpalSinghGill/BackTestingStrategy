# Gold daily candle cut — CoinDCX / TradingView XAUUSDT

Copy everything below the line into the other Cursor chat (or `@` this file).

---

## Task

Build gold **day candles from 1-minute data**. Match **CoinDCX XAUUSDT** and **TradingView XAUUSDT** day windows.

Do **not** cut at IST midnight (12:00 AM). That is what the current 1m files look like if you group by calendar date. It is **wrong** for CoinDCX / XAUUSDT.

## Data already on disk

- Folder: `GOLD_DATA/Yahoo_Finance/` (Yahoo COMEX) or `GOLD_DATA/TradingView_Vantage/` (Vantage XAUUSD)
- 1m files: `August_2026.csv`, `September_2026.csv`, … (month name + year)
- Columns: `Datetime,Open,High,Low,Close,Volume`
- `Datetime` is **Asia/Kolkata IST**, offset `+05:30` (example: `2026-09-16 05:30:00+05:30`)
- Source symbol is COMEX `GC=F`. Time window must match XAUUSDT; OHLC will not match CoinDCX prices exactly.

`Gold_Daily.csv` is COMEX/US session dates. Do **not** use it if the goal is CoinDCX / XAUUSDT day candles.

## Correct day window (UTC day, shown in IST)

Each daily candle is one **UTC day** = **00:00 UTC → 23:59 UTC**.

In IST that is:

- **Start (include):** `05:30:00` IST
- **End (include):** next calendar day `05:29:00` IST
- **Next day starts at:** next calendar day `05:30:00` IST (do not put that bar in the previous day)

Example — Monday daily candle:

- Include Monday `05:30` IST through Tuesday `05:29` IST
- Tuesday `05:30` IST opens Tuesday’s daily

Label the daily bar with the **UTC date**, which is the same as the IST date of the `05:30` open (Monday 05:30 IST = Monday 00:00 UTC → label Monday).

## How to bucket 1m bars

```text
session_date = (Datetime_IST - 5 hours 30 minutes).date()
```

Equivalent: convert IST → UTC, then take the UTC calendar date.

OHLC for that session_date:

- Open = first 1m Open in the window
- High = max 1m High
- Low = min 1m Low
- Close = last 1m Close
- Volume = sum of 1m Volume

Skip empty days (weekends / COMEX halt) — do not invent bars.

## Do not use these cuts

- IST `00:00` → `23:59` (calendar IST day) — **wrong**
- COMEX `18:00` US Eastern → next day `17:00` ET (~03:30–02:30 IST in summer) — **wrong for CoinDCX**
- Forex XAUUSD New York close `17:00` ET — **wrong for CoinDCX**

CoinDCX charts are TradingView of **XAUUSDT**. Same cut as crypto UTC daily. Other TradingView gold symbols (COMEX `GC1!`, forex `XAUUSD`) use different cuts; ignore those.

## Output

Save day candles in the same feed subfolder, e.g. `GOLD_DATA/TradingView_Vantage/Gold_Daily_XAUUSDT_UTC.csv` or `GOLD_DATA/Yahoo_Finance/Gold_Daily_XAUUSDT_UTC.csv`, with `Date,Open,High,Low,Close,Volume`. `Date` = UTC session date (`YYYY-MM-DD`). Keep existing month 1m files unchanged.
