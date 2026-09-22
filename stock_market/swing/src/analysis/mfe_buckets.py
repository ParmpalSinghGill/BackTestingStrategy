from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "backtest_trades.csv")
m = df["mfe_pct"]
n = len(df)

print(f"Total trades: {n}")
print(f"Avg MFE% (all trades): {m.mean():.2f}%\n")

print(f"{'threshold':<10}{'count':>7}{'% of all':>10}{'avg MFE% of those':>20}")
for t in (1, 2, 3):
    hit = m >= t
    avg = m[hit].mean() if hit.any() else 0.0
    print(f">= {t}%{'':<5}{hit.sum():>7}{100*hit.mean():>9.0f}%{avg:>19.2f}%")

print(f"\nReached < 1%: {(m < 1).sum()} ({100*(m < 1).mean():.0f}%)")

print("\n--- by side ---")
for side, g in df.groupby("side"):
    ms = g["mfe_pct"]
    print(f"{side:<6} n={len(g):>3}  avg MFE% {ms.mean():.2f}  "
          f">=1%: {(ms>=1).sum()}  >=2%: {(ms>=2).sum()}  >=3%: {(ms>=3).sum()}")
