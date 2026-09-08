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
* 📅 **Daily entries (4 PM)**: `python swing_strategy/run_daily_all_forecasts.py` — download once, then `Swing_Live.txt` (calendar Y/M/W lows), `Swing_low.txt`, and `Swing_PP.txt`
* 📓 **Tried experiments (do not retry)**: [strategy/TRIED_EXPERIMENTS.md](strategy/TRIED_EXPERIMENTS.md)

> 🔒 **Git Repository Storage Policy**: Only source code, documentation, and build scripts are tracked in Git. All historical datasets (`data/`, `data_daily/`), generated statements (`Reports/`), and chart graphics (`Plots/`) are ignored via `.gitignore`.

---

## 📁 Repository Structure

```
.
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
├── README.md                     # Base Swing Strategy Documentation
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
