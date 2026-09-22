# Session State Restoration: Stock Backtesting Strategy

Relocation notes plus the **current verified strategy**. Live rules and returns: [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md).

---

## 1. Project Path Relocation Checklist

When you move this project to its new directory path:

### A. Update `run_fetch.bat`
The batch wrapper script contains a hardcoded absolute path. Edit it in the new location:
```batch
@echo off
cd /d [YOUR_NEW_PROJECT_PATH]
C:\Users\parmp\anaconda3\python.exe fetch_1min_data.py
```

### B. Update Windows Task Scheduler Task
The Windows Task Scheduler task `StockBacktest_Fetch1m` runs `run_fetch.bat` every Saturday at 9:00 AM. After moving:
```powershell
schtasks /change /tn "StockBacktest_Fetch1m" /tr "[YOUR_NEW_PROJECT_PATH]\run_fetch.bat"
```

---

## 2. Current Strategy (verified 5 Sep 2026)

Not the old 15-minute reversal and not the no-ML scaled TF→Nifty book. Live spec: [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md). Do-not-retry: [strategy/TRIED_EXPERIMENTS.md](strategy/TRIED_EXPERIMENTS.md).

- Liquidity: on sweep, if next support below is within 5%, use the lower level
- C1: green, **open below** liquidity
- C2 / A1: C2 close > C1 high before C1 low breaks; enter **C3 at C3 open**
- SL: sweep-to-C1 lowest low × 0.99
- Exit: full position at 1:2
- Selection: walk-forward XGBoost + meta-label; drop `Meta_P < 0.40`; daily top 16 by `Meta_P`
- Sizing: Moreira–Muir `clip(0.04 / day_median_idio_vol, 0.40, 1.80)` × rank 1.4× → 0.6×
- Same code for ₹50k/₹500, ₹100k/₹500, ₹100k/₹1k
- Runner: `python swing_strategy/run_ml_wave4.py`

---

## 3. Verified benchmark (replace any older “official” record)

Net Zerodha CAGR 2010-01-01 to 2026-08-28 (walk-forward, Indian tax):

| Book | Net CAGR | Max DD |
|------|----------|--------|
| ₹50,000 / ₹500 | **+34.87%** | 37.4% |
| ₹100,000 / ₹500 | **+30.72%** | 21.9% |
| ₹100,000 / ₹1,000 | **+35.24%** | 36.5% |

Previous no-ML baseline (₹50k/₹1k, 33/33/34 @ 1:1/1:3/1:4): ₹152,212 / +6.91% / 78% DD.

**Discarded numbers (do not restore as current):** Dynamic ML +35,840%; README ML +870% / +1,952% / +2,322%; C1-submerged / Pure / Confirmed / SwingNoMl (negative after tax); oracle lookahead top-N; ₹2,800–₹4,000 risk 40% prints (not the three target books).
