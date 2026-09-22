"""
Trend Finder & Market Regime Classifier Engine

Provides quantitative methods for market trend identification:
1. Supertrend (ATR-based dynamic trailing trend filter)
2. ADX & DMI (+DI / -DI trend strength & direction)
3. EMA Alignment & Slope (Fast/Medium/Slow EMA confluence & angle)
4. Pivot Price Action (Higher Highs / Higher Lows / Lower Highs / Lower Lows)
5. Rolling Linear Regression (Trend slope m & linearity R^2)
6. Composite Market Regime Classification (Strong Uptrend, Weak Uptrend, Ranging, Weak Downtrend, Strong Downtrend)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend line and trend signal (+1 for Uptrend, -1 for Downtrend).
    Returns (supertrend_series, trend_direction_series).
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    atr = calculate_atr(df, period=period).values
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.zeros(n, dtype=int)
    st = np.zeros(n)

    for i in range(1, n):
        # Final Upper Band logic
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band logic
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Trend direction logic
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                st[i] = final_upper[i]
            else:
                trend[i] = 1
                st[i] = final_lower[i]
        elif trend[i - 1] == -1:
            if close[i] > final_upper[i]:
                trend[i] = 1
                st[i] = final_lower[i]
            else:
                trend[i] = -1
                st[i] = final_upper[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                st[i] = final_lower[i]
            else:
                trend[i] = -1
                st[i] = final_upper[i]

    return pd.Series(st, index=df.index, name="Supertrend"), pd.Series(
        trend, index=df.index, name="Supertrend_Signal"
    )


def calculate_adx(
    df: pd.DataFrame, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI.
    Returns (adx, plus_di, minus_di).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_smoothed = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100.0 * (plus_dm_smoothed / tr_smoothed.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smoothed / tr_smoothed.replace(0, np.nan))

    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan))

    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return (
        adx.fillna(0.0).rename("ADX"),
        plus_di.fillna(0.0).rename("Plus_DI"),
        minus_di.fillna(0.0).rename("Minus_DI"),
    )


def calculate_ema_features(
    df: pd.DataFrame, fast_p: int = 20, med_p: int = 50, slow_p: int = 200, slope_lookback: int = 5
) -> Dict[str, pd.Series]:
    """Calculate EMAs, alignment, and slope angle."""
    close = df["Close"]
    ema_fast = close.ewm(span=fast_p, adjust=False).mean()
    ema_med = close.ewm(span=med_p, adjust=False).mean()
    ema_slow = close.ewm(span=slow_p, adjust=False).mean()

    # Slope in percentage terms per bar
    ema_fast_slope_pct = (
        (ema_fast - ema_fast.shift(slope_lookback)) / ema_fast.shift(slope_lookback)
    ) * (100.0 / slope_lookback)

    # EMA Slope Angle (degrees)
    slope_angle = np.degrees(np.arctan(ema_fast_slope_pct / 100.0))

    # EMA Alignment: 1 if fast > med > slow, -1 if fast < med < slow, 0 otherwise
    alignment = np.where(
        (ema_fast > ema_med) & (ema_med > ema_slow),
        1,
        np.where((ema_fast < ema_med) & (ema_med < ema_slow), -1, 0),
    )

    return {
        "EMA_Fast": ema_fast,
        "EMA_Med": ema_med,
        "EMA_Slow": ema_slow,
        "EMA_Fast_Slope_Pct": ema_fast_slope_pct.fillna(0.0),
        "EMA_Fast_Slope_Angle": pd.Series(slope_angle, index=df.index).fillna(0.0),
        "EMA_Alignment": pd.Series(alignment, index=df.index),
    }


def calculate_pivot_structure(
    df: pd.DataFrame, window: int = 3, lookback_pivots: int = 4
) -> Dict[str, Any]:
    """
    Detect pivot highs and pivot lows, and determine price action structure (HH/HL or LH/LL).
    Returns dict with pivot classifications and overall structural bias (+1 Bullish, -1 Bearish, 0 Ranging).
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    pivot_highs = []
    pivot_lows = []

    for i in range(window, n - window):
        # Pivot High: current high strictly greater than surrounding window
        if all(high[i] >= high[i - k] for k in range(1, window + 1)) and all(
            high[i] >= high[i + k] for k in range(1, window + 1)
        ):
            pivot_highs.append((i, high[i]))

        # Pivot Low: current low strictly lower than surrounding window
        if all(low[i] <= low[i - k] for k in range(1, window + 1)) and all(
            low[i] <= low[i + k] for k in range(1, window + 1)
        ):
            pivot_lows.append((i, low[i]))

    # Analyze last few pivots
    recent_ph = pivot_highs[-lookback_pivots:] if len(pivot_highs) >= lookback_pivots else pivot_highs
    recent_pl = pivot_lows[-lookback_pivots:] if len(pivot_lows) >= lookback_pivots else pivot_lows

    higher_highs = 0
    lower_highs = 0
    for k in range(1, len(recent_ph)):
        if recent_ph[k][1] > recent_ph[k - 1][1]:
            higher_highs += 1
        elif recent_ph[k][1] < recent_ph[k - 1][1]:
            lower_highs += 1

    higher_lows = 0
    lower_lows = 0
    for k in range(1, len(recent_pl)):
        if recent_pl[k][1] > recent_pl[k - 1][1]:
            higher_lows += 1
        elif recent_pl[k][1] < recent_pl[k - 1][1]:
            lower_lows += 1

    if higher_highs > lower_highs and higher_lows > lower_lows:
        bias = 1  # Bullish HH + HL structure
    elif lower_highs > higher_highs and lower_lows > higher_lows:
        bias = -1  # Bearish LH + LL structure
    else:
        bias = 0  # Mixed / Ranging structure

    return {
        "Pivot_Highs_Count": len(pivot_highs),
        "Pivot_Lows_Count": len(pivot_lows),
        "Higher_Highs_Count": higher_highs,
        "Higher_Lows_Count": higher_lows,
        "Lower_Highs_Count": lower_highs,
        "Lower_Lows_Count": lower_lows,
        "Pivot_Structure_Bias": bias,
    }


def calculate_linear_regression_trend(
    series: pd.Series, window: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate rolling linear regression slope and R-squared (R^2) linearity metric.
    Returns (slope, r_squared).
    """
    n = len(series)
    slopes = np.zeros(n)
    r2s = np.zeros(n)

    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    vals = series.values

    for i in range(window - 1, n):
        y = vals[i - window + 1 : i + 1]
        if np.isnan(y).any():
            continue

        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        slope = cov / x_var if x_var != 0 else 0.0

        # Percentage slope normalized by current price
        slope_pct = (slope / y[-1]) * 100.0 if y[-1] != 0 else 0.0
        slopes[i] = slope_pct

        # Calculate R^2
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2s[i] = max(0.0, min(1.0, r2))

    return pd.Series(slopes, index=series.index, name="LinReg_Slope_Pct"), pd.Series(
        r2s, index=series.index, name="LinReg_R2"
    )


class TrendFinder:
    """
    Comprehensive Quantitative Trend Finder & Market Regime Classifier.
    Integrates Supertrend, ADX, DMI, EMA alignment/slope, Pivot Price Action, and Linear Regression.
    """

    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        adx_period: int = 14,
        ema_fast: int = 20,
        ema_med: int = 50,
        ema_slow: int = 200,
        linreg_window: int = 20,
    ):
        self.st_period = supertrend_period
        self.st_mult = supertrend_multiplier
        self.adx_period = adx_period
        self.ema_fast = ema_fast
        self.ema_med = ema_med
        self.ema_slow = ema_slow
        self.linreg_window = linreg_window

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all technical indicators and appends market regime metrics to dataframe.
        """
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain required columns: {required_cols}")

        res = df.copy()

        # 1. Supertrend
        st_line, st_signal = calculate_supertrend(
            res, period=self.st_period, multiplier=self.st_mult
        )
        res["Supertrend"] = st_line
        res["Supertrend_Signal"] = st_signal

        # 2. ADX & DMI
        adx, plus_di, minus_di = calculate_adx(res, period=self.adx_period)
        res["ADX"] = adx
        res["Plus_DI"] = plus_di
        res["Minus_DI"] = minus_di

        # 3. EMA Features
        ema_dict = calculate_ema_features(
            res, fast_p=self.ema_fast, med_p=self.ema_med, slow_p=self.ema_slow
        )
        for k, v in ema_dict.items():
            res[k] = v

        # 4. Rolling Linear Regression
        lr_slope, lr_r2 = calculate_linear_regression_trend(
            res["Close"], window=self.linreg_window
        )
        res["LinReg_Slope_Pct"] = lr_slope
        res["LinReg_R2"] = lr_r2

        # 5. Composite Regime & Confidence Score
        regime_labels = []
        regime_scores = []
        confidence_pcts = []

        st_sig = res["Supertrend_Signal"].values
        adx_val = res["ADX"].values
        p_di = res["Plus_DI"].values
        m_di = res["Minus_DI"].values
        ema_align = res["EMA_Alignment"].values
        ema_angle = res["EMA_Fast_Slope_Angle"].values
        lr_m = res["LinReg_Slope_Pct"].values

        n = len(res)
        for i in range(n):
            score = 0
            max_points = 5

            # Supertrend contribution (+1 / -1)
            score += st_sig[i]

            # DMI Direction contribution
            if p_di[i] > m_di[i]:
                score += 1
            elif m_di[i] > p_di[i]:
                score -= 1

            # EMA Alignment contribution
            score += ema_align[i]

            # EMA Slope direction
            if ema_angle[i] > 1.0:
                score += 1
            elif ema_angle[i] < -1.0:
                score -= 1

            # Linear Regression Slope
            if lr_m[i] > 0.05:
                score += 1
            elif lr_m[i] < -0.05:
                score -= 1

            # Calculate confidence level based on indicator agreement & ADX strength
            trend_strength_mult = min(1.5, max(0.5, adx_val[i] / 25.0)) if adx_val[i] > 0 else 0.5
            conf = min(100.0, round((abs(score) / max_points) * 100.0 * trend_strength_mult, 1))

            # Regime Label
            if score >= 3 and adx_val[i] >= 20:
                label = "STRONG_UPTREND"
            elif score >= 1:
                label = "WEAK_UPTREND"
            elif score <= -3 and adx_val[i] >= 20:
                label = "STRONG_DOWNTREND"
            elif score <= -1:
                label = "WEAK_DOWNTREND"
            else:
                label = "RANGING"

            regime_labels.append(label)
            regime_scores.append(score)
            confidence_pcts.append(conf)

        res["Market_Regime"] = regime_labels
        res["Regime_Score"] = regime_scores
        res["Trend_Confidence_Pct"] = confidence_pcts

        return res

    def get_latest_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs analysis and returns a clean dictionary summary of the latest bar's trend diagnostics.
        """
        analyzed_df = self.analyze(df)
        pivot_info = calculate_pivot_structure(df)

        last_row = analyzed_df.iloc[-1]
        summary = {
            "Date": str(last_row.get("Date", "Latest")),
            "Close": float(last_row["Close"]),
            "Market_Regime": str(last_row["Market_Regime"]),
            "Regime_Score": int(last_row["Regime_Score"]),
            "Trend_Confidence_Pct": float(last_row["Trend_Confidence_Pct"]),
            "Supertrend_Signal": "Bullish" if last_row["Supertrend_Signal"] == 1 else "Bearish",
            "Supertrend_Value": float(last_row["Supertrend"]),
            "ADX": float(last_row["ADX"]),
            "Trend_Strength": "Strong" if last_row["ADX"] >= 25 else ("Moderate" if last_row["ADX"] >= 20 else "Weak/Ranging"),
            "Plus_DI": float(last_row["Plus_DI"]),
            "Minus_DI": float(last_row["Minus_DI"]),
            "EMA_Fast_Slope_Angle": float(last_row["EMA_Fast_Slope_Angle"]),
            "EMA_Alignment": "Bullish Stack (20>50>200)" if last_row["EMA_Alignment"] == 1 else ("Bearish Stack (20<50<200)" if last_row["EMA_Alignment"] == -1 else "Mixed"),
            "LinReg_Slope_Pct": float(last_row["LinReg_Slope_Pct"]),
            "LinReg_R2": float(last_row["LinReg_R2"]),
            "Pivot_Structure": "Higher Highs & Higher Lows" if pivot_info["Pivot_Structure_Bias"] == 1 else ("Lower Highs & Lower Lows" if pivot_info["Pivot_Structure_Bias"] == -1 else "Consolidation / Mixed"),
            "Higher_Highs": pivot_info["Higher_Highs_Count"],
            "Higher_Lows": pivot_info["Higher_Lows_Count"],
            "Lower_Highs": pivot_info["Lower_Highs_Count"],
            "Lower_Lows": pivot_info["Lower_Lows_Count"],
        }
        return summary
