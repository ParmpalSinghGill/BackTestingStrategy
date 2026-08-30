# Specification Guide: Machine Learning Input Features & Target Labels Dictionary

This document provides a complete technical dictionary of all **Input Features ($X$)** and **Target Labels ($y$)** used across the Quantitative Machine Learning Strategy Engine (`swing_strategy/`).

---

## 📌 Data Architecture & Zero-Lookahead Isolation

To guarantee institutional-grade backtesting integrity:
1. **Input Features ($X$)**: Contain **strictly pre-entry historical data** derived from candles prior to the trade entry ($C1$ and $C2$). The model has **zero access** to post-entry price movement during training or prediction.
2. **Target Labels ($y$)**: Derived from historical forward simulations (`Max_RR_Achieved`). Used **only** as the training target ($y$) during `clf.fit(X_train, y_train)` and for exit evaluation during portfolio backtesting.

```
       PRE-ENTRY FEATURES (X)                 ENTRY (C3)           POST-ENTRY EVALUATION (y)
[Candles -5 to C2] -> Pre-Sweep Indicators  ===> BUY AT C3 OPEN ===> Max_RR_Achieved & Exit_Date_MaxRR
```

---

## 📊 1. Input Features Dictionary ($X$)

The feature matrix $X$ passed to Machine Learning classifiers (`RandomForestClassifier`, `LightGBM`) consists of **23 technical and price-action features**:

### A. Stock Liquidity & Market Index Context Features
| Feature Name | Type | Unit / Range | Technical Description |
| :--- | :--- | :--- | :--- |
| `Nifty_Rank` | Categorical | 1 to 4 | Market cap tier ranking of ticker (`4`: Nifty 50, `3`: Nifty 100, `2`: Nifty 250, `1`: Other/Microcap). |
| `Support_Type_Rank` | Categorical | 1 to 3 | Higher timeframe liquidity support level strength (`3`: Yearly Level, `2`: Monthly Level, `1`: Weekly Level). |
| `Sweep_Depth_Pct` | Continuous | Percentage (%) | Percentage distance the lowest wick penetrated below key support before reversing. |
| `Pre_Sweep_Runup_Pct` | Continuous | Percentage (%) | Percentage price advance preceding the liquidity sweep pullback. |
| `Red_Candles_Before_C1` | Discrete | 1 to 10+ | Number of consecutive red (bearish) candles forming the pullback leading into candle $C1$. |
| `Intermediary_Candles_Count` | Discrete | 0 to 20 | Number of candles between the initial support level touch and signal candle $C1$. |

---

### B. Signal Candle ($C1$) Price-Action & Volatility Features
| Feature Name | Type | Unit / Range | Technical Description |
| :--- | :--- | :--- | :--- |
| `C1_Pattern_Rank` | Categorical | 1 to 5 | Reversal candle pattern classification (`Engulfing`, `Hammer`, `Pinbar`, `Doji`, etc.). |
| `C1_Body_Pct` | Continuous | 0.0 to 1.0 | Ratio of candle real body size to total high-low candle range ($\frac{|\text{Close} - \text{Open}|}{\text{High} - \text{Low}}$). |
| `C1_Upper_Wick_Pct` | Continuous | 0.0 to 1.0 | Ratio of upper wick length to total high-low candle range. |
| `C1_Lower_Wick_Pct` | Continuous | 0.0 to 1.0 | Ratio of lower wick length (reversal rejection wick) to total high-low candle range. |
| `C1_Range_Pct` | Continuous | Percentage (%) | High-to-Low candle range normalized by Close price ($\frac{\text{High} - \text{Low}}{\text{Close}} \times 100$). |
| `ATR20_Pct` | Continuous | Percentage (%) | 20-period Average True Range (ATR) as a percentage of Close price ($\frac{\text{ATR}_{20}}{\text{Close}} \times 100$). |
| `Dist_SMA50_Pct` | Continuous | Percentage (%) | Percentage distance between signal Close price and the 50-day Simple Moving Average ($\frac{\text{Close} - \text{SMA}_{50}}{\text{SMA}_{50}} \times 100$). |

---

### C. Preceding Price-Action Context Features (`prev_1` to `prev_5`)
For each of the 5 candles preceding $C1$ (`prev_1` = candle immediately before $C1$, `prev_5` = 5 candles before $C1$):

| Feature Name Pattern | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `prev_N_color` | Binary | 0 or 1 | Candle direction (`1`: Green/Bullish, `0`: Red/Bearish). |
| `prev_N_body_pct` | Continuous | 0.0 to 1.0 | Real body ratio of historical candle $N$. |
| `prev_N_upper_wick_pct` | Continuous | 0.0 to 1.0 | Upper wick ratio of historical candle $N$. |
| `prev_N_lower_wick_pct` | Continuous | 0.0 to 1.0 | Lower wick ratio of historical candle $N$. |

---

## 🎯 2. Target Labels Dictionary ($y$)

Target labels are derived from forward historical simulation by tracking max price expansion before hitting Stop-Loss ($R = \text{Entry Price} - \text{SL Price}$):

| Label Feature Name | Class Value | Target Name | Assigning Condition | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Four_Class_Label` | **`0`** | `Skip` | $\text{Max RR} < 2.0$ or $\text{Outcome} = \text{Fail}$ | Trade failed to reach 1:2 RR or hit Stop-Loss first. |
| | **`1`** | `1:2 RR` | $2.0 \le \text{Max RR} < 3.0$ | Reached base 1:2 Risk-Reward target. |
| | **`2`** | `1:3 RR` | $3.0 \le \text{Max RR} < 4.0$ | Reached high-tier 1:3 Risk-Reward target. |
| | **`3`** | `1:4 RR` | $\text{Max RR} \ge 4.0$ | Reached momentum breakout 1:4+ Risk-Reward target. |
| `Five_Class_Label` | **`4`** | `1:5 RR` | $5.0 \le \text{Max RR} < 6.0$ | Reached 1:5 Risk-Reward target. |
| `Six_Class_Label` | **`5`** | `1:6 RR` | $\text{Max RR} \ge 6.0$ | Reached 1:6+ Risk-Reward runner target. |

---

## 🔍 Ground-Truth Groundwork Columns

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `Max_RR_Achieved` | Continuous / Discrete | Maximum multiple of risk ($R$) reached by the stock price **before** hitting Stop-Loss ($0, 2, 3, 4, 5, 6, 10, 15$). |
| `Exit_Date_MaxRR` | YYYY-MM-DD Date | Historical calendar date when `Max_RR_Achieved` was hit. Used by simulation engine for exit date calculation. |
