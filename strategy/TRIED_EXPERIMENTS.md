# Tried experiments — do not retry

Log of finished tests. **Do not re-run these** unless the universe, label, or engine changed. Add a new row when you finish a test.

Last updated: 9 Sep 2026

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
| M36 | Swing-low HTF liquidity (`LIQUIDITY.md`): N=3 / N2=3, wick-sweeps ≤ M2, close-below kills level; same C1/C2/ML | Best M2=1 **+12.43 / +13.36 / +14.08**, DD 55 / 51 / 55. M2=0 **+6.02 / +11.77 / +12.21**. M2=2 **+14.54 / +11.79 / +14.96**. Universe ~29k vs 159k | Don’t promote; don’t re-grid M2 on this N/N2 |
| M37 | Swing-low 2010-01-01 to 2016-12-31, ML vs NoML, 6 capital/risk books | Best ML **+16.84%** (M2=0, ₹200k/₹500). Live three books ML all **< +11%**. NoML mostly ≤ +4%, several negative | Don’t promote; 2010–2016 does not rescue swing-low |
| M38 | Swing-low 2/2 both sides + 3 on one side; skip 2 dailies after source; confirm after 2 right HTF bars (3 if left only has 2); same C1/C2/ML | Best M2=2 **+23.34 / +19.99 / +23.84**, DD 39 / 35 / 39. M2=0 **+22.59 / +19.81 / +23.13**. M2=1 **+17.93 / +17.31 / +18.63** with 47–58% DD. Universe ~37k vs 159k | Don’t promote; don’t re-grid M2 on this neighbour rule |
| M39 | Same as M38, window 2010-01-01 to 2016-12-31, ML vs NoML, 6 books | Best live-three ML M2=2 **+32.47 / +30.49 / +33.11**. Peak ML **+42.39%** (M2=2, ₹50k/₹1k). NoML still ~0% / negative | Don’t promote; short window is not the live book |
| M40 | Swing-low **or** neighbour rule: 2 on both sides **or** 3 on one side; always ≥2 after the low (immediate next HTF bar is never a trade bar); skip 2 dailies after source; same C1/C2/ML | Best M2=0 **+21.33 / +20.21 / +22.04**, DD 46 / 42 / 44. M2=2 **+21.27 / +19.68 / +21.96**. M2=1 **+19.96 / +19.63 / +21.02** with 45–58% DD. Universe ~43–44k vs 159k | Don’t promote; looser than M38 and worse CAGR |
| M41 | Locked swing-low **AND** rule (2 both + 3 one, M2=2). Hunt: C1 OPEN/CLOSE/HIGH_BELOW × A1/A2/A3/B/C2-close × 1:2/1:3; XGB/HGB/MLP/CatBoost meta; frozen risk ₹500–₹10k; **% of equity** 1–10%. No future features. | Best tradable **+47.19%** net (HGB meta t0.42 top32, **2% of current equity**, vol 0.04, 1,568 trades, 51.9% DD, ₹50k start). Frozen-rupee peak **+43.51%** at ₹8,000 (still <45). Oracle lookahead ceiling **+37.6%** at ₹500 / **+43.0%** at ₹1k / **+54.4%** at ₹4k. C2-close and 1:3 lost to OPEN_BELOW+A1 1:2 after ML | 45% needs **% of equity** (or huge frozen rupee). Do not cite 47% as a ₹500 book. Don’t re-grid this exact HGB/t0.42/top32/2% combo |
| M42 | Same 2% equity HGB book: after K consecutive closed losses, pause D days or skip next T setups (then resume). Not a permanent halt (that was M24). | After 3+ losses, next-trade WR falls 30%→21%. Best DD-keep-CAGR: pause 5 SL / 20d **+44.25% / 36.5% DD**; skip 2 SL / 10 trades **+45.50% / 36.8%**. Best grid print: skip 5 SL / 5 trades **+52.01% / 40.5% DD** (DD moves to 2024-09-20→2025-03-20). Aggressive pauses crush CAGR | Don’t treat the +52% skip as live without a hold-out; grid was fit on the full sample |
| M42b | Wider search on the same 2% HGB book: skip Feb, rolling WR, DD circuit, size-cut, same-day multi-loss pause, K×D combos | Best both: after **6 SLs pause 15d** **+52.79% / 38.1% DD**. Best DD keep ~47%: **2 losses same day → pause 10d** **+47.32% / 33.8% DD**. Skip-Feb + skip5 **+58.71%** but DD **56.3%** (worse). Rolling-WR and DD-circuit gates often killed CAGR | Don’t promote skip-February; seasonal fit. Prefer 6-SL/15d or same-day-2-loss/10d |
| M43 | Same 2% HGB book, but gate **real fills** using **paper** 1:2/SL of trades not taken. Shadow all 37,485 setups or ML-only 10,903. Paper result applied only after Exit_Date (next session). Consecutive paper SLs, rolling paper WR, last-N SL ratio. | Consecutive paper SLs on **all** over-pause (CAGR ~5–34%). Best keep-CAGR: last **20 all-setup** paper exits, pause while SL ratio ≥75% **+47.64% / 36.9% DD** (n=1,020; DD 2018-05-07→2019-06-21). Best both: last **20 ML paper** WR <22% **+54.16% / 37.6% DD** (n=1,152; DD 2024-09-20→2025-05-07). Peak CAGR: last **30 ML** SL≥75% **+61.29% / 46.4% DD** (still 2010–11 trough). `ml_roll20_0.22`, `ml_roll20_0.25`, `ml_slratio20_0.8` are the **same discrete rule** (4/20 wins). | Don’t treat +54/+61 as live; full-sample fit. Don’t re-grid this exact paper window. Prefer SL-ratio over consecutive-K on the all-setup shadow |
| M44 | Same swing-low C1/A1/HGB/2% book, but **only old panic-point liquidity**: HTF swing low must be ≥1 / ≥2 / ≥3 calendar months before Entry_Date. Fresh weekly lower-lows dropped. Same-scores filter vs HGB+meta retrain. | Fresh &lt;1m is 969 setups, **almost all Weekly**, WR 37% / mean R 0.12 vs ≥3m WR 39% / 0.19 R. **Retrain ≥1m: 2% book +49.43% / 34.7% DD** (n=1,486) vs baseline +47.19% / 51.9%. Same-scores ≥1m +45.25% / 50.2% DD. ≥2m and ≥3m kill CAGR (~+27% / +26% retrain; frozen ₹500 ~+17 / +14). Frozen 1m still ~+23%, below live calendar lows. | Don’t promote 2m/3m. 1m retrain DD cut is full-sample; don’t put in Swing_low.txt yet. Don’t re-grid age on this exact HGB book |
| M45 | ≥2m liquidity + **Nifty-scaled min volume** (no fixed share count): C2 turnover/Nifty, shares/Nifty, turnover/Nifty², plus vol_ratio20 / rel_dolvol floors. Then Meta_P / top-N / % equity / HGB+XGB retrain / paper gates / filled-SL pause / DD circuit. Target 50% CAGR and &lt;25% DD. | Volume floors on default t0.42/top32/2% **hurt** vs ≥2m baseline +29%/50% DD. Lift is **shares ≥ 0.487×Nifty** (15th pctile; ~2.4k shares at Nifty 5k, ~12k at 25k) + **Meta_P≥0.48**. 2% equity **+45.05% / 34.4% DD**. 4% equity **+51.07% / 49.9% DD**. After 6 filled SLs pause 15d: **+53.22% / 30.3% DD** (n=565). DD circuit at 25–30% wipes CAGR (~−3%). Frozen ₹500 **+13.5%**. **No book hit 50% CAGR and DD&lt;25.** Floor for 50% CAGR is ~30% DD. | Don’t put in Swing_low.txt. Don’t cite 53% as a ₹500 book. Don’t re-grid this exact sh/Nifty×t0.48×4%×pause6/15 combo. 25% DD is incompatible with 50% CAGR on the ≥2m book |
| M46 | Locked **2%** equity (not 4%). Fair daily-bar BE: until +1R, SL is checked first on the bar; after 1R, hunt 1:2 then scratch at entry. ≥1m HGB retrain, Meta_P≥0.48 top 16. Vol-scale cap **1.3** (default 1.8), rank cap **1.2** (default 1.4). If **2 filled losses the same day**, pause **8** days. | **+53.40% / 24.5% DD**, n=1,360, ₹50k start. Same-bar MFE-only BE was invalid (optimistic +57%/24% on ≥2m collapsed to ~+8% SL-first). ≥2m at 2% still cannot print 50% CAGR and DD&lt;25 together without the invalid BE. | Full-sample pause. Do not cite as ₹500. Not in Swing_low.txt. Don’t treat MFE-only BE as live. Don’t re-grid this exact 1.3/1.2/dl2_8 stack on this book |
| M47 | Swing_PP paper-gate 1:2 book: walk-forward XGB RR picker. At-entry vs after-+1R (raise TP). Labels = fair-BE 1:3/1:4/1:5 win. Paper-gate stays 1:2. Pre-specified stretch bar: P≥0.80 (need >67% to beat banking 2R). | Unconditional 1:3 **+41.4% / 32.9% DD**. At-entry P never clears ~57% precision — do not pick RR at C3 open. **After +1R, P(1:3)≥0.80 → 1:3 else 1:2: +59.94% / 24.5% DD**, n=1,324, 86 higher-RR fills. OOS precision **86.3%** (2011–18 86.0%; 2019–26 86.9%). t=0.90 **+60.70% / 24.5%** (threshold peeked). Scale-out 50/50 weaker than full-size 1:3 on the rare flags. Oracle lookahead 1:4-if-win4 **+96.7% / 20.7%** (ceiling). | Not a ₹500 book. Do not put in Swing_low / Swing_Live. Do not always-1:3. Do not use at-entry RR. Don’t re-grid this exact t on the same book; t=0.80 is the economic cut |
| M48 | Same Swing_PP 1R switch, add 1:5 / 1:6. Walk-forward P(win_5/win_6) after +1R. Stack on M47 t3=0.80. Economic vs 2R: 1:5>40%, 1:6>33%; vs already-1:3: 1:5>60%, 1:6>50%. | **DD-flat stack** P6≥0.85→1:6 elif P5≥0.85→1:5 elif P3≥0.80→1:3: **+63.29% / 24.5% DD**, fills 16×1:6 + 5×1:5 + 65×1:3. Fast-arm (≤3 bars) P6≥0.80: **+64.10% / 24.5%**, 23×1:6. Looser t6=0.75: +65.13% / **24.6% DD**. OOS prec 1:5 t=0.85 **75%** (n=36); 1:6 t=0.85 **64%** (n=25). Oracle 6-else-3 **+136% / 20.7%** (lookahead). Rules on close_R overfit vs ML stack. | Not ₹500. Not Swing_low/Live. Don’t always-1:5/1:6. Don’t loosen t6 below 0.85 if DD must stay 24.5 |

---

## Papers already used (don’t re-read as if new)

- Gu, Kelly, Xiu (RFS 2020) — trees; momentum / liquidity / volatility
- Poh et al. (2020) Learning to Rank — LambdaMART / NDCG@K
- López de Prado AFML — purge labels, drop low-gain features, **meta-labeling** (M18)
- Feature design note (C1/C2 geometry) — implemented in v6
- Joubert (JFDS 2022) Meta-Labeling: Theory and Framework — take/skip + size from P(win)
- Moreira & Muir (JF 2017) Volatility-Managed Portfolios — scale risk when recent vol is high (M21)
- CatBoost (Prokhorenkova et al., 2018) — ordered boosting; tried M41, no beat of HGB
- Fixed-fraction / percent-of-equity size (Vince / standard futures sizing) — M41 2% of current equity

---

## Not tried yet (allowed next)

- Torch listwise / set ranker over the day’s candidates
- Full TrendFinder 5/20/50/100d pack (M30 only added ADX/Supertrend/EMA/linreg20)
