# Tried experiments — do not retry

Log of finished tests. **Do not re-run these** unless the universe, label, or engine changed. Add a new row when you finish a test.

Last updated: 6 Sep 2026

---

## Setup / entry / SL (no ML)

| ID | What | Result | Do not retry |
|---|---|---|---|
| T01 | Strict C1 fully submerged (close **and** high below), 1:3, SL=C1 low | All red after tax (e.g. ₹100k/₹500 → −69%) | Yes |
| T02 | Confirmed C2/C3 Yearly/Monthly 1:2 | −50% to −69% net | Yes |
| T03 | SwingNoMl (SL support×0.999, C2-only 1:3) | Account wiped | Yes |
| T04 | User-Defined Pure / C1-submerged rescan 3 Sep 2026 | −6.9% to −10% net CAGR | Yes |
| T05 | Scaled 1:1/1:3/1:4, close-below, Touch 1.001, SL=C1 low, TF→Nifty, ₹50k/₹1k | **+6.91%** net, 78% DD, 942 trades | Keep as baseline only |
| T06 | 48-variant C1×entry×SL MFE matrix | Best count: OPEN_BELOW + A1 + SWEEP_x99 (15.1% hit 1:10) | Matrix itself done |
| T07 | 2nd attempt if SL before 1:2; last SL = next liquidity; 1:2 and 1:3 full exit | All 6 books red after tax; att2 net negative | Yes |
| T08 | Oracle lookahead: rank same-day by realized R, top 10 random/greedy | ₹50k/500 1:2 random → ~₹1.83 Cr, +42.5% CAGR (lookahead) | Ceiling only |
| T09 | Oracle top 5 greedy/random | Greedy 1:3 ₹50k/500 → ₹2.00 Cr, +43.3% CAGR (lookahead) | Ceiling only |

---

## ML / DL selection on OPEN_BELOW+A1+SWEEP_x99 (1:2 unless noted)

Walk-forward yearly, features before entry + today’s open. Universe ~172k setups.

| ID | What | Best net CAGR (50k/500 unless noted) | Do not retry |
|---|---|---|---|
| M01 | LGBM/XGB/HGB/MLP classify `y_win`, daily top 1/3/5 ± p≥0.50–0.65 | XGB top5 ₹50k/500 **+24.98%**; thresholds cut CAGR | Yes |
| M02 | LGBM/XGB regress clipped realized R, top 5/3/1 | ≤ +24.0% | Yes |
| M03 | LGBM binary `y_top5` (in oracle top 5) | Poor (₹50k/500 +7.8% at top5) | Yes |
| M04 | Ensemble of XGB/LGBM/HGB/reg scores, top 5–12 | Top12 ₹50k/500 +26.4%; ₹100k/500 +22.6% | Yes |
| M05 | LGBM LambdaRank (int R labels), expanding + roll-6 | ~+24–25% | Yes |
| M06 | Recency-weighted XGB/LGBM | No lift vs M01 | Yes |
| M07 | Universe filters: Yearly+Monthly only, Nifty 50/100 only, both | All worse (CAGR 4–13%) | Yes |
| M08 | Train on 1:3 labels, full 1:3 exit, XGB/LGBM top 5/8 | ₹50k/500 ~+26.9%; WR ~34% | Yes |
| M09 | RR3 ensemble top 8–20 | Same ~27% ceiling at ₹500 | Yes |
| M10 | Risk ₹1,000 on ensemble/idio (not 500/1k target set) | ₹50k/1k +32.7% (idio top12) | Different risk; don’t re-grid 1k on 50k for “same 500 books” |
| M11 | Risk ₹1,200–₹4,000 grid, idio top 10–14, ₹50k | 40%+ from **₹2,800** up; best +42.65% at ₹4,000 top11 | Do not re-grid high risk for 40% |
| M12 | v6 features: A1 geometry + GKU mom/vol/dolvol + CS ranks; prune TF/Nifty/sweep_age; purge overlapping exits | XGB+rank top12: 50k/500 **+27.2%**, 100k/500 **+23.6%**, 100k/1k **+28.3%** | Yes |
| M13 | LambdaRank with within-day quantile relevance (Poh et al.) on v6 | No beat of XGB | Yes |
| M14 | n_cands ≥8 or ≥10; risk_pct band 2–12% | Worse than unfiltered | Yes |
| M15 | Train `y_top5` on v6 features | No beat | Yes |
| M16 | Dynamic 1:2 vs 1:3 exit from predicted MFE (cuts 1.6–3.0) | Best min-CAGR still ~23.5% on 100k/500 | Yes |
| M17 | Score-weighted size (1.4× → 0.6× risk) top 8/12/16 | 50k/500 **+28.6%**, 100k/500 **+24.8%**, 100k/1k **+29.7%** (top16) | Beaten by M18; don’t re-tune this exact taper |
| M18 | Meta-label (2nd XGB on v6 + primary score): take/skip thresh 0.40/0.48/0.52; rank ML_Score or Meta_P; linear & Kelly size | Kelly top16 t0.40 **+31.39 / +26.73 / +31.82**. Filter t0.48+ too tight | Beaten by M25; don’t re-grid these exact cuts |
| M19 | Scaled 33/33/34 @ 1:1/1:3/1:4 on ML top-16 A1 (flat and score-sized) | Flat 50k/500 **−1.76%**; sized min **+20.5%**. Worse than full 1:2 | Yes |
| M20 | Monthly (not yearly) walk-forward XGB, top 12/16, score-sized | Best top16 **+30.20 / +26.45 / +31.35**. No beat of M18 min | Yes |
| M21 | Moreira–Muir vol-managed size on XGB top16 (scale risk by target / day-median idio_vol) | tv0.03 **+31.82 / +29.02 / +32.91** (50k DD 44%) | Beaten by M25; don’t re-grid these three targets |
| M24 | Skip new entries after K consecutive closed losses (k=3/4/6) on meta top16 | k=3 executed ~12 trades, ~0% CAGR (gate stuck shut) | Yes |
| M22 | Vertical time barrier exit at close if 1:2/SL not hit in 5/10/20 bars | All worse (h5 min +17%, h20 min +22%) | Yes |
| M25 | Meta filter (t0.40, top16 by Meta_P) + Moreira–Muir vol size tv 0.02/0.03/0.04 | tv0.04 **+34.87 / +30.72 / +35.24** | Beaten by M27 top20; don’t re-grid these three tv values |
| M26 | Monthly primary + yearly meta + vol tv0.04 top16 | +34.31 / +29.89 / +34.68 (lower DD ~18%) | Yes |
| M23 | Meta-label on 1:3 exits, top20 t0.40, Kelly and vol | Kelly min +22.9%; vol min +26.3%. Worse than 1:2 | Yes |
| M27 | New gates: t0.38 top16 tv0.04/0.05; t0.40 top20 tv0.04 | top20 **+35.53 / +31.36 / +35.91** | Beaten by M28 top24 |
| M28 | Combos t0.38/0.36/0.42 top20; t0.40 top24; all vol 0.04 | top24 **+35.94 / +31.70 / +36.31** | Beaten by M31 |
| M29 | Tail-quality (i≥12 & P<0.45; i≥8 & P<0.48), cash-rich P≥0.45, day spend cap 25/35% | All worse than ungated top24 (daycap crashed to ~+24%) | Yes |
| M30 | ADX/DI + Supertrend + EMA align/slope + 20d linreg added to v6; yearly XGB+meta+vol top24 | +35.20 / +30.70 / +35.63 | Yes |
| M31 | t0.38 top24; t0.40 top28; t0.38 top28; vol 0.04 | t0.38 top28 **+36.34 / +32.27 / +36.89** | Beaten by M33 top32 |
| M32 | Per-name vol scale (not day-median) on top24 t0.40 | +36.00 / +31.73 / +36.36, better DD | Yes |
| M33 | t0.36 top28; t0.38 top32; t0.36 top32 | t0.38 top32 **+36.52 / +32.50 / +37.03** on the *old* universe (reused swept doji highs). Axis saturating | Don’t repeat; don’t go top40 on this axis |
| M34 | Per-name vol on live t0.38 top32 | +36.55 / +32.49 / +37.01. Tie on min, better DD | Yes |
| M35 | Bugfix: skip flat/doji HTF prints; 5% rule uses **intact** supports only (no reused swept lows / upper liquidity) | **Live:** t0.38 top32 vol **+36.39 / +31.38 / +36.81**, DD 32.4 / 28.6 / 30.8. Universe 159k vs 172k | Don’t reintroduce swept-level substitution |

---

## Papers already used (don’t re-read as if new)

- Gu, Kelly, Xiu (RFS 2020) — trees; momentum / liquidity / volatility
- Poh et al. (2020) Learning to Rank — LambdaMART / NDCG@K
- López de Prado AFML — purge labels, drop low-gain features, **meta-labeling** (M18)
- Feature design note (C1/C2 geometry) — implemented in v6
- Joubert (JFDS 2022) Meta-Labeling: Theory and Framework — take/skip + size from P(win)
- Moreira & Muir (JF 2017) Volatility-Managed Portfolios — scale risk when recent vol is high (M21)

---

## Not tried yet (allowed next)

- CatBoost walk-forward (package not installed)
- ML ranker on the old close-below TouchPlannedEntry trade list
- Other matrix entries (A2, A3, B) with ML — only A1 was ranked
- Torch listwise / set ranker over the day’s candidates
- Full TrendFinder 5/20/50/100d pack (M30 only added ADX/Supertrend/EMA/linreg20)
