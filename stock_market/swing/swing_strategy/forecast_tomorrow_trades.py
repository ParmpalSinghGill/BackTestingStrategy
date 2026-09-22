"""
Automated Stock Forecast Engine for Swing Strategy (TRUE LIQUIDITY SWEEP & PENDING C3 SETUPS)

Strict Setup Rules:
1. True Liquidity Sweep: C1 Low MUST strictly penetrate BELOW Support Price (C1 Low < Support Price, Sweep Depth > 0.0%).
2. Reversal Confirmation (C1): Green Candle (Close > Open).
3. Pre-Sweep Trend: Preceded by at least 1 Red Candle in previous 3 bars.
4. Pending C3 Entry: C1 is the VERY LAST COMPLETED CANDLE in the stock file (C3 HAS NOT FORMED YET).
5. High-Confidence ML Filter: Selects top setups with Probability Score P >= 45.0%.

Outputs Generated:
- forecast_stocks/ & latest_fyers/ Swing_Stocks.xlsx & Swing_31_Aug_2026.txt
"""

import os
import sys
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from sklearn.ensemble import RandomForestClassifier

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"
DATA_DAILY_DIR = BASE_DIR / "data_daily"
FORECAST_DIR = BASE_DIR / "forecast_stocks"
FYERS_DIR = BASE_DIR / "latest_fyers"

FORECAST_DIR.mkdir(parents=True, exist_ok=True)
FYERS_DIR.mkdir(parents=True, exist_ok=True)

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.backtest_engine.backtest_support_liquidity_strategy import INDEX_CLASSIFIER, get_all_stock_supports


def to_fyers_symbol(ticker: str) -> str:
    clean = ticker.replace(".NS", "").replace(".BO", "").replace("_NS", "").replace("_BO", "").split("=")[0].upper()
    if ".BO" in ticker or "_BO" in ticker:
        return f"BSE:{clean}-EQ"
    return f"NSE:{clean}-EQ"


def get_last_completed_trading_date(override_now=None) -> datetime.date:
    now = override_now or datetime.datetime.now()
    weekday = now.weekday()
    current_time = now.time()
    
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    
    is_inside_market = (weekday < 5) and (market_open <= current_time < market_close)
    today = now.date()
    
    if is_inside_market:
        target_date = today - datetime.timedelta(days=1)
        while target_date.weekday() >= 5:
            target_date -= datetime.timedelta(days=1)
        print(f"[Market Hours Check] Current time ({now.strftime('%H:%M:%S')}) is INSIDE market hours. Considering YESTERDAY ({target_date}) as last completed day.", flush=True)
        return target_date
    else:
        target_date = today
        if weekday == 5:
            target_date = today - datetime.timedelta(days=1)
        elif weekday == 6:
            target_date = today - datetime.timedelta(days=2)
        elif current_time < market_open:
            target_date = today - datetime.timedelta(days=1)
            while target_date.weekday() >= 5:
                target_date -= datetime.timedelta(days=1)
                
        print(f"[Market Hours Check] Current time ({now.strftime('%H:%M:%S')}) is AFTER market close / weekend. Considering TODAY ({target_date}) as last completed day.", flush=True)
        return target_date


def train_ml_model(training_cutoff_str: str = "2016-01-01"):
    dataset_path = REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv"
    if not dataset_path.exists():
        dataset_path = REPORTS_DIR / "Four_Class_Trade_Features_Dataset.csv"
        
    if not dataset_path.exists():
        print("Warning: Base dataset not found for ML training.", flush=True)
        return None, None

    df = pd.read_csv(dataset_path)
    
    if "Four_Class_Label" not in df.columns:
        def assign_4class_label(row):
            outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
            max_rr = float(row.get("Max_RR_Achieved", 0.0))
            if outcome == "Fail" or max_rr < 2.0:
                return 0
            elif max_rr >= 4.0:
                return 3
            elif max_rr >= 3.0:
                return 2
            else:
                return 1
        df["Four_Class_Label"] = df.apply(assign_4class_label, axis=1)

    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    
    sc_col = "Scenario" if "Scenario" in df.columns else "Scenario_1to2"
    df_sc1 = df[df[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
    
    cutoff_dt = pd.to_datetime(training_cutoff_str)
    train_df = df_sc1[df_sc1["C2_Date"] < cutoff_dt]
    
    if len(train_df) < 100:
        train_df = df_sc1

    base_feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
    feat_cols = [c for c in base_feat_cols if c in df_sc1.columns]

    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df["Four_Class_Label"].values

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"ML Model trained on {len(train_df):,} historical setups across {len(feat_cols)} features.", flush=True)
    return clf, feat_cols


def process_single_stock_forecast(csv_file: Path, last_dt_ts: pd.Timestamp, clf, feat_cols):
    ticker = csv_file.stem.replace("_1d", "").replace("_NS", ".NS").replace("_BO", ".BO")
    if "ticker_cache" in ticker:
        return None, None

    try:
        df = pd.read_csv(csv_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return None, None

    if len(df) < 30:
        return None, None

    # C1 MUST BE THE VERY LAST COMPLETED CANDLE IN THE STOCK FILE!
    # (If C3 already exists in the file, setup played out in the past and MUST BE EXCLUDED)
    n = len(df)
    c1_idx = n - 1

    c1_open = float(df.at[c1_idx, "Open"])
    c1_high = float(df.at[c1_idx, "High"])
    c1_low = float(df.at[c1_idx, "Low"])
    c1_close = float(df.at[c1_idx, "Close"])
    c1_date = df.at[c1_idx, "Date"]
    c1_date_str = c1_date.strftime("%Y-%m-%d")

    # STRICT RULE 1: C1 Must be a Green Reversal Candle
    if c1_close <= c1_open:
        return None, None

    # STRICT RULE 2: Must be preceded by at least 1 Red Candle in previous 3 bars
    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values

    red_count = 0
    for k in range(c1_idx - 1, max(-1, c1_idx - 4), -1):
        if closes[k] < opens[k]:
            red_count += 1
        else:
            break

    if red_count < 1:
        return None, None

    # STRICT RULE 3: Support formed STRICTLY BEFORE C1 Date
    df_history = df.iloc[:c1_idx]
    all_supports = get_all_stock_supports(df_history.set_index("Date"))
    if not all_supports:
        return None, None

    # STRICT RULE 4: TRUE LIQUIDITY PENETRATION! C1 Low MUST strictly penetrate BELOW Support Price!
    strict_swept_supports = []
    for s in all_supports:
        sup_p = s["price"]
        if c1_low < sup_p and c1_low >= sup_p * 0.93:
            strict_swept_supports.append(s)

    if not strict_swept_supports:
        return None, None

    best_sup = min(strict_swept_supports, key=lambda s: abs(s["price"] - c1_low))
    sup_price = best_sup["price"]
    tf_label = best_sup["timeframe"]

    sweep_depth_pct = ((sup_price - c1_low) / sup_price) * 100.0

    NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
    TIMEFRAME_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}

    idx_tag = INDEX_CLASSIFIER.classify(ticker)
    nifty_rank_val = NIFTY_RANK.get(idx_tag, 1)
    tf_rank_val = TIMEFRAME_RANK.get(tf_label, 1)

    c1_body = abs(c1_close - c1_open)
    c1_range = max(0.01, c1_high - c1_low)
    lower_wick = min(c1_open, c1_close) - c1_low

    if c1_body / c1_range >= 0.7:
        pattern_name = "Marubozu"
    elif lower_wick / c1_range >= 0.5:
        pattern_name = "Hammer / Pinbar"
    elif c1_idx > 0 and closes[c1_idx-1] < opens[c1_idx-1] and c1_close > opens[c1_idx-1]:
        pattern_name = "Bullish Engulfing"
    else:
        pattern_name = "Standard Green Reversal"

    entry_p = c1_high
    sl_p = c1_low
    risk_per_share = max(0.1, entry_p - sl_p)

    target_1to2 = round(entry_p + 2.0 * risk_per_share, 2)
    target_1to3 = round(entry_p + 3.0 * risk_per_share, 2)
    target_1to4 = round(entry_p + 4.0 * risk_per_share, 2)

    max_risk_cap = 1000.0
    suggested_qty = max(1, int(max_risk_cap / risk_per_share))
    est_investment = round(suggested_qty * entry_p, 2)

    fyers_sym = to_fyers_symbol(ticker)

    no_ml_info = {
        "Fyers_Symbol": fyers_sym,
        "Signal_Date (C1)": c1_date_str,
        "Forecast_Entry_Date (C3)": (c1_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "Ticker": ticker,
        "Index_Membership": idx_tag,
        "Nifty_Rank": nifty_rank_val,
        "Support_Type": tf_label,
        "Support_Price": round(sup_price, 2),
        "Sweep_Depth_Pct": round(sweep_depth_pct, 2),
        "Pattern_Name": pattern_name,
        "Entry_Price": round(entry_p, 2),
        "SL_Price": round(sl_p, 2),
        "Risk_Per_Share": round(risk_per_share, 2),
        "Target_Price (1:2)": target_1to2,
        "Suggested_Qty (1k Risk)": suggested_qty,
        "Est_Capital_Required": est_investment,
        "Setup_Status": "TRUE LIQUIDITY SWEEP (C3 Pending)"
    }

    ml_info = None

    if clf is not None and feat_cols is not None:
        c1_body_pct = c1_body / c1_range
        c1_upper_wick_pct = (c1_high - max(c1_open, c1_close)) / c1_range
        c1_lower_wick_pct = lower_wick / c1_range
        c1_range_pct = (c1_range / c1_close) * 100.0

        pre_high_idx = max(0, c1_idx - 20)
        pre_sweep_high = np.max(highs[pre_high_idx:c1_idx]) if c1_idx > pre_high_idx else highs[c1_idx]
        pre_sweep_runup_pct = ((pre_sweep_high - sup_price) / sup_price) * 100.0 if sup_price > 0 else 0.0

        atr20_pct = 2.5
        dist_sma50_pct = 0.0
        if n >= 50:
            sma50 = np.mean(closes[n-50:n])
            dist_sma50_pct = ((c1_close - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0

        feat_dict = {
            "Nifty_Rank": nifty_rank_val,
            "Support_Type_Rank": tf_rank_val,
            "Sweep_Depth_Pct": sweep_depth_pct,
            "Pre_Sweep_Runup_Pct": pre_sweep_runup_pct,
            "Red_Candles_Before_C1": red_count,
            "Intermediary_Candles_Count": 2,
            "C1_Pattern_Rank": 3,
            "C1_Body_Pct": c1_body_pct,
            "C1_Upper_Wick_Pct": c1_upper_wick_pct,
            "C1_Lower_Wick_Pct": c1_lower_wick_pct,
            "C1_Range_Pct": c1_range_pct,
            "ATR20_Pct": atr20_pct,
            "Dist_SMA50_Pct": dist_sma50_pct,
        }

        for prev_num in range(1, 6):
            p_idx = c1_idx - prev_num
            c_label = f"prev_{prev_num}"
            if p_idx >= 0:
                po, ph, pl, pc = opens[p_idx], highs[p_idx], lows[p_idx], closes[p_idx]
                prng = max(0.01, ph - pl)
                feat_dict[f"{c_label}_color"] = 1 if pc > po else 0
                feat_dict[f"{c_label}_body_pct"] = abs(pc - po) / prng
                feat_dict[f"{c_label}_upper_wick_pct"] = (ph - max(po, pc)) / prng
                feat_dict[f"{c_label}_lower_wick_pct"] = (min(po, pc) - pl) / prng
            else:
                feat_dict[f"{c_label}_color"] = 0
                feat_dict[f"{c_label}_body_pct"] = 0.5
                feat_dict[f"{c_label}_upper_wick_pct"] = 0.25
                feat_dict[f"{c_label}_lower_wick_pct"] = 0.25

        row_feats = [feat_dict.get(c.replace("_1to2", ""), 0.0) for c in feat_cols]
        X_curr = pd.DataFrame([row_feats], columns=feat_cols).fillna(0)

        probs = clf.predict_proba(X_curr)[0]
        classes = clf.classes_

        p0 = probs[np.where(classes == 0)[0][0]] if 0 in classes else 0.0
        p1 = probs[np.where(classes == 1)[0][0]] if 1 in classes else 0.0
        p2 = probs[np.where(classes == 2)[0][0]] if 2 in classes else 0.0
        p3 = probs[np.where(classes == 3)[0][0]] if 3 in classes else 0.0

        total_enter_prob = p1 + p2 + p3

        # STRICT ML CONFIDENCE FILTER (Select High-Confidence Setups P >= 0.45)
        if total_enter_prob >= 0.45:
            if p3 >= 0.40:
                rec_target = "1:4"
                t_price = target_1to4
            elif p2 >= 0.40:
                rec_target = "1:3"
                t_price = target_1to3
            else:
                rec_target = "1:2"
                t_price = target_1to2

            ml_info = no_ml_info.copy()
            ml_info["Recommended_Target_RR"] = rec_target
            ml_info["Recommended_Target_Price"] = t_price
            ml_info["ML_Confidence_Prob"] = round(total_enter_prob * 100.0, 1)

    return no_ml_info, ml_info


def scan_all_stocks_fast(last_completed_date: datetime.date, clf, feat_cols):
    print(f"\nScanning daily stock charts in {DATA_DAILY_DIR} (Enforcing TRUE Liquidity Sweep C1 Low < Support Price & C3 Pending)...", flush=True)
    csv_files = list(DATA_DAILY_DIR.glob("*.csv"))
    last_dt_ts = pd.Timestamp(last_completed_date)

    no_ml_candidates = []
    ml_candidates = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_stock_forecast, f, last_dt_ts, clf, feat_cols) for f in csv_files]
        for fut in as_completed(futures):
            res_no_ml, res_ml = fut.result()
            if res_no_ml:
                no_ml_candidates.append(res_no_ml)
            if res_ml:
                ml_candidates.append(res_ml)

    return no_ml_candidates, ml_candidates


def format_and_save_excel(df: pd.DataFrame, excel_path: Path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "True Swing Forecast"
    
    ws.views.sheetView[0].showGridLines = True
    headers = list(df.columns)
    ws.append(headers)
    
    header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    
    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )
    
    for col_num in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_num)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
    row_fill_even = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    row_fill_odd = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    
    for r_idx, row in df.iterrows():
        row_num = r_idx + 2
        ws.append(list(row))
        fill = row_fill_even if r_idx % 2 == 0 else row_fill_odd
        
        for c_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=row_num, column=c_idx)
            cell.fill = fill
            cell.border = thin_border
            cell.font = Font(name="Calibri", size=10)
            
            if c_idx in [1, 2, 3, 4, 5, 6, 7, 10, 14]:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="right", vertical="center")
                
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 15)
        
    wb.save(excel_path)
    print(f"Exported formatted Excel file to: {excel_path.resolve()}", flush=True)


def export_fyers_text_file(df_master: pd.DataFrame, today_date: datetime.date):
    base_indices = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    stock_syms = df_master["Fyers_Symbol"].dropna().tolist() if "Fyers_Symbol" in df_master.columns else []

    seen = set()
    unique_stocks = []
    for s in stock_syms:
        if s not in seen and "NONE" not in s:
            seen.add(s)
            unique_stocks.append(s)

    full_symbol_list = base_indices + unique_stocks
    text_content = ",".join(full_symbol_list)

    date_str = today_date.strftime("%d_%b_%Y")
    txt_filename = f"Swing_{date_str}.txt"

    out_forecast = FORECAST_DIR / txt_filename
    out_fyers = FYERS_DIR / txt_filename
    out_root = BASE_DIR / txt_filename

    for out_p in [out_forecast, out_fyers, out_root]:
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(text_content)
            
    print(f"Exported FYERS text list file ({len(full_symbol_list)} symbols) -> {txt_filename}", flush=True)
    return txt_filename


def main():
    print("==========================================================================", flush=True)
    print("  SWING FORECAST ENGINE (TRUE LIQUIDITY SWEEP C1 LOW < SUPPORT & C3 PENDING) ", flush=True)
    print("==========================================================================", flush=True)

    last_completed_date = get_last_completed_trading_date()
    print(f"• Last Completed Trading Date Considered: {last_completed_date}", flush=True)

    clf, feat_cols = train_ml_model(training_cutoff_str="2016-01-01")

    no_ml_list, ml_list = scan_all_stocks_fast(last_completed_date, clf, feat_cols)

    df_no_ml = pd.DataFrame(no_ml_list)
    df_ml = pd.DataFrame(ml_list)

    print(f"\nScan Completed!")
    print(f"• Total True Liquidity Sweep Setups (WITHOUT ML): {len(df_no_ml):,}")
    print(f"• Total High-Confidence ML Setups (WITH ML):       {len(df_ml):,}")

    file_swing_master_xlsx = FORECAST_DIR / "Swing_Stocks.xlsx"
    file_swing_master_csv = FORECAST_DIR / "Swing_Stocks.csv"

    file_swing_ml_xlsx = FORECAST_DIR / "Swing_Stocks_with_ML.xlsx"
    file_swing_ml_csv = FORECAST_DIR / "Swing_Stocks_with_ML.csv"

    file_swing_no_ml_xlsx = FORECAST_DIR / "Swing_Stocks_without_ML.xlsx"
    file_swing_no_ml_csv = FORECAST_DIR / "Swing_Stocks_without_ML.csv"

    fyers_swing_xlsx = FYERS_DIR / "Swing_Stocks.xlsx"
    fyers_swing_csv = FYERS_DIR / "Swing_Stocks.csv"

    if not df_ml.empty:
        df_master = df_ml.sort_values(["ML_Confidence_Prob", "Nifty_Rank"], ascending=[False, False]).reset_index(drop=True)
    elif not df_no_ml.empty:
        df_master = df_no_ml.sort_values(["Nifty_Rank", "Sweep_Depth_Pct"], ascending=[False, False]).reset_index(drop=True)
    else:
        df_master = pd.DataFrame([{
            "Fyers_Symbol": "NSE:NONE-EQ",
            "Signal_Date (C1)": str(last_completed_date),
            "Forecast_Entry_Date (C3)": str(last_completed_date + datetime.timedelta(days=1)),
            "Ticker": "NONE",
            "Setup_Status": "No Setups Found"
        }])

    df_master.to_csv(file_swing_master_csv, index=False)
    format_and_save_excel(df_master, file_swing_master_xlsx)

    df_master.to_csv(fyers_swing_csv, index=False)
    format_and_save_excel(df_master, fyers_swing_xlsx)

    if not df_ml.empty:
        df_ml = df_ml.sort_values(["ML_Confidence_Prob", "Nifty_Rank"], ascending=[False, False]).reset_index(drop=True)
        df_ml.to_csv(file_swing_ml_csv, index=False)
        format_and_save_excel(df_ml, file_swing_ml_xlsx)

    if not df_no_ml.empty:
        df_no_ml = df_no_ml.sort_values(["Nifty_Rank", "Sweep_Depth_Pct"], ascending=[False, False]).reset_index(drop=True)
        df_no_ml.to_csv(file_swing_no_ml_csv, index=False)
        format_and_save_excel(df_no_ml, file_swing_no_ml_xlsx)

    txt_filename = export_fyers_text_file(df_master, datetime.date.today())

    print("\n==========================================================================")
    print("      TRUE SWING STOCKS FORECAST FILES GENERATED SUCCESSFULLY            ")
    print("==========================================================================")
    print(f"1. FYERS Single-Line Text File: {txt_filename} (in forecast_stocks/, latest_fyers/ & root)")
    print(f"2. Master Swing Stocks Excel:   {file_swing_master_xlsx.resolve()}")
    print(f"3. ML Filtered Swing Excel:     {file_swing_ml_xlsx.resolve()}")
    print("==========================================================================\n", flush=True)


if __name__ == "__main__":
    main()
