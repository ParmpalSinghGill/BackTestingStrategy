# Swing Trading Strategy Suite (Base Repository)

Quantitative swing trading for Indian equities (NSE/BSE). **Current verified best** is walk-forward XGBoost + meta-label + Moreira-Muir vol-managed 1:2 size on OPEN_BELOW + A1 setups. See [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md). Do-not-retry log: [strategy/TRIED_EXPERIMENTS.md](strategy/TRIED_EXPERIMENTS.md).

---

## 🔗 Cross-Repository Navigation & Guides

* 📊 **Base Strategy Repository**: **Swing Trading Strategy Suite** (`swing_strategy/`)
* ⚡ **Subfolder Module**: [Intraday Trading Strategy Suite](intraday_strategy/README.md) (`intraday_strategy/`)
* 📘 **LLM Realism Specification Guide**: [Guide/Realistic_guide.md](Guide/Realistic_guide.md) (Master 8-Rule Prompt for AI Agents)
* 📊 **Account Statement & MTM Equity Guide**: [Guide/Account_Statement_guide.md](Guide/Account_Statement_guide.md) (21-Column Institutional Schema & MTM Valuation Rules)
* 📈 **Output Format & Tax Impact Guide**: [Guide/OutputFormatGuide.md](Guide/OutputFormatGuide.md) (Working PNG Hyperlinks, Visualizations & Before/After Tax CAGR Engine)
* 🏆 **Current Best Strategy**: [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md) (verified 5 Sep 2026 rules and net CAGRs)
* ⚙️ **Algorithm (what it does)**: [strategy/ALGORITHM.md](strategy/ALGORITHM.md)
* 📅 **Daily entries (4 PM)**: `python swing_strategy/run_daily_all_forecasts.py` — download once, then `Swing_Live.txt`, `Swing_low.txt`, `Swing_PP.txt` (enter next open; scratch if that close is below C1 high), `Swing_PP_RR.txt` (SHIFT target FROM 1:2 TO 1:k after a +1R close), and always-on `Swing_PP_ins.txt`
* 📓 **Tried experiments (do not retry)**: [strategy/TRIED_EXPERIMENTS.md](strategy/TRIED_EXPERIMENTS.md)
* 🥇 **Gold / silver replay chart**: [`gold_chart/`](gold_chart/) — local GUI at http://127.0.0.1:8766 (see [run latest](#run-the-latest-replay-chart))

> 🔒 **Git Repository Storage Policy**: Only source code, documentation, and build scripts are tracked in Git. Historical datasets (`data/`, `data_daily/`, `MARKET_DATA/`), generated statements (`Reports/`), and chart graphics (`Plots/`) are ignored. Replay-chart screenshots in `gold_chart/docs/` are tracked so this README can show the live GUI.

---

## Run the latest replay chart

GitHub: [ParmpalSinghGill/BackTestingStrategy](https://github.com/ParmpalSinghGill/BackTestingStrategy)

### First clone

```bash
git clone https://github.com/ParmpalSinghGill/BackTestingStrategy.git
cd BackTestingStrategy
```

### Already cloned — pull the latest `main`

```bash
git checkout main
git pull origin main
```

That is the chart you should run: paper Isolated book, SL/TP that only fill on a later touch, chart partial exits, Events Min (Hourly/Daily/Weekly/Monthly), and the Data dropdown for COMEX gold, Vantage gold spot, and Vantage silver spot.

### Start the GUI

Needs Python 3 and `pandas` (`pip install -r requirements.txt`). The chart **does not download** market data; it reads local `MARKET_DATA/` folders.

```bash
python gold_chart/app.py
```

Then open **http://127.0.0.1:8766/index.htm**. If the page looks like an old build, hard-refresh (Ctrl+F5) or add `?v=62` (current `app.js` cache).

On Windows you can also double-click `run_gold_chart.bat`. `run_gold_chart_server.vbs` starts the same server hidden (used at sign-in so the chart is already up).

If the port is already serving, `app.py` prints that and exits — just open the URL above.

### Data feeds (local only)

| Data dropdown | Folder | Symbol |
|---|---|---|
| COMEX Gold Futures | `MARKET_DATA/Yahoo_Finance_Gold` | `GC=F` |
| Vantage Gold Spot | `MARKET_DATA/TradingView_Vantage_Gold` | `XAUUSD` |
| Vantage Silver Spot | `MARKET_DATA/TradingView_Vantage_Silver` | `XAGUSD` |

1m/intraday uses recent month CSVs; 1D/1W uses `Gold_Daily.csv` / `Silver_Daily.csv`. The hidden 6-hour fetch job updates those folders; the GUI never fetches Yahoo or TradingView itself.

### What the GUI does

![Gold replay chart — COMEX 1h with paper ticket](gold_chart/docs/gold-replay-gui.png)

* **Replay** — pick Date + Time IST, then Play / Next / Next hour / Next day. Drawings stay in the browser.
* **Paper** — Isolated, CoinDCX-style taker + GST + slippage. Drag TP/SL on the chart (short SL above mark, long SL below). Partial close from the ticket or 25/50/75% chips. Bought vs remaining qty and mark value stay in the ticket; P&L chips stay pinned on the right.
* **Events** — optional S/R labels. **Min** (Hourly default) is the finest TF for Next/Prev and on-chart labels; coarser timeframes stay on.

![Gold replay chart — Events on, Min Hourly](gold_chart/docs/gold-replay-events.png)

---

## 📁 Repository Structure

```
.
├── gold_chart/                   # Local gold/silver replay GUI (port 8766)
│   ├── app.py
│   ├── paper.py
│   ├── static/
│   └── docs/                     # Live GUI screenshots used above
├── run_gold_chart.bat            # Open browser + start app.py
├── swing_strategy/
│   ├── run_ml_wave4.py           # Current best: meta-label + vol-managed 1:2
│   ├── run_ml_next_search.py     # Meta-label Kelly (no vol scale)
│   ├── run_ml_sized.py           # Score-weighted XGB only
│   ├── run_scaled_1_3_4_strategy.py  # Previous no-ML baseline
│   ├── tiered_liquidity_strategy_engine.py
│   ├── generate_statement.py
│   ├── visualizer.py
│   └── plotter.py
├── strategy/
│   ├── BEST_STRATEGY.md          # Verified rules + net CAGRs
│   └── TRIED_EXPERIMENTS.md      # Finished tests — do not retry
├── Guide/
│   ├── Realistic_guide.md
│   ├── Account_Statement_guide.md
│   └── OutputFormatGuide.md
├── intraday_strategy/            # Intraday Trading Strategy Suite (See intraday_strategy/README.md)
├── src/                          # Shared brokerage tax & fee calculators
│   └── analysis/
│       └── indian_brokerage_calculator.py
├── .gitignore                    # Code-only git rules (excluding datasets, plots & reports)
├── README.md
└── requirements.txt              # Python package dependencies
```

---

## 🚀 Current Best Run

```bash
python swing_strategy/run_ml_wave4.py
```

- Same selector for every book; only capital and risk change.
- Rules and **verified** net CAGRs: [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md).
- Last verified net (Zerodha, 2010–2026): **₹50k/₹500 +36.52%**, **₹100k/₹500 +32.50%**, **₹100k/₹1k +37.03%** (meta-label top 32, `Meta_P ≥ 0.38`, vol-managed size).
- Previous no-ML baseline (₹50k/₹1k scaled 1:1/1:3/1:4): +6.91% net, 78% DD.

---

## 📈 Swing Strategy Benchmarks

* **Entries-First Execution Rule**: New trade entries process FIRST at market open using available cash at the start of Day $D$. Day $D$ exits process SECOND, releasing cash for Day $D+1$ onwards.
* **Active Position Count Tracking**: Includes an explicit `Active Position Count` column across every statement transaction row.
