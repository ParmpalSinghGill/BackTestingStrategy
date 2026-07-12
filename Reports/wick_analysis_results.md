# Reversal Candle Wick Filter Analysis (With Sequential Re-entries)

This report displays the portfolio return results for different timeframes and reversal candle wick filters, combining both the **09:30 AM Skip Filter** and the **Sequential Re-entry rule** (up to 3 tries on immediate SL hit).

## Timeframe: 1m

| Wick Combination          |   Candidate Trades |   Accepted Trades |   Net Return % |
|:--------------------------|-------------------:|------------------:|---------------:|
| Default (No limit)        |               1081 |               125 |      -17.9199  |
| 1:2 (1% side, 2% other)   |               1044 |               116 |      -57.7036  |
| 1:5 (1% side, 5% other)   |               1019 |               121 |      -52.6671  |
| 2:5 (2% side, 5% other)   |               1019 |               121 |      -52.6671  |
| 2:10 (2% side, 10% other) |               1029 |               104 |       -3.68495 |
| 5:10 (5% side, 10% other) |               1027 |               102 |       -3.70802 |
| 5:15 (5% side, 15% other) |               1055 |               118 |      -22.2177  |

## Timeframe: 5m

| Wick Combination          |   Candidate Trades |   Accepted Trades |   Net Return % |
|:--------------------------|-------------------:|------------------:|---------------:|
| Default (No limit)        |                404 |                50 |        4.67113 |
| 1:2 (1% side, 2% other)   |                215 |                20 |       -4.46081 |
| 1:5 (1% side, 5% other)   |                237 |                22 |       -6.70153 |
| 2:5 (2% side, 5% other)   |                242 |                24 |       -1.29527 |
| 2:10 (2% side, 10% other) |                265 |                30 |       -3.28025 |
| 5:10 (5% side, 10% other) |                280 |                42 |      -17.5682  |
| 5:15 (5% side, 15% other) |                307 |                45 |      -18.236   |

## Timeframe: 10m

| Wick Combination          |   Candidate Trades |   Accepted Trades |   Net Return % |
|:--------------------------|-------------------:|------------------:|---------------:|
| Default (No limit)        |                211 |                42 |       -8.94443 |
| 1:2 (1% side, 2% other)   |                 42 |                18 |       -8.33918 |
| 1:5 (1% side, 5% other)   |                 46 |                16 |      -19.9619  |
| 2:5 (2% side, 5% other)   |                 48 |                16 |      -19.9619  |
| 2:10 (2% side, 10% other) |                 60 |                17 |      -17.7695  |
| 5:10 (5% side, 10% other) |                 84 |                22 |      -34.4824  |
| 5:15 (5% side, 15% other) |                 99 |                25 |      -29.7947  |

## Timeframe: 15m

| Wick Combination          |   Candidate Trades |   Accepted Trades |   Net Return % |
|:--------------------------|-------------------:|------------------:|---------------:|
| Default (No limit)        |                135 |                25 |        16.6485 |
| 1:2 (1% side, 2% other)   |                 29 |                13 |       -23.3862 |
| 1:5 (1% side, 5% other)   |                 30 |                12 |       -30.4351 |
| 2:5 (2% side, 5% other)   |                 34 |                12 |       -29.877  |
| 2:10 (2% side, 10% other) |                 36 |                13 |       -22.3289 |
| 5:10 (5% side, 10% other) |                 54 |                20 |       -21.8537 |
| 5:15 (5% side, 15% other) |                 58 |                20 |       -23.6726 |

