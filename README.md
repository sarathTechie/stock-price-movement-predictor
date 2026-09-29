# Stock Price Movement Predictor

This project evaluates a stock movement model using Indian equities, historical OHLCV data, walk-forward out-of-fold (OOF) predictions, and a daily-bar backtest.

## Setup

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data and model workflow

```powershell
python src\feature_engineering.py
python src\generate_oof_predictions.py
python src\backtest.py
```

## Running tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -p "*.py"
```

## Smoke checks

```powershell
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m src.backtest
```

## Interpretation

- Entries occur at the next trading session Open after the signal date closes.
- Exit logic uses daily High/Low barriers and a conservative same-day ambiguity rule.
- If both target and stop are touched on the same daily bar, the stop is treated as the exit and the trade is flagged as ambiguous.
- Time exits use the Close of the final allowed holding session when no barrier is touched.
- Transaction costs are treated as assumptions and are configurable in [src/backtest.py](src/backtest.py).

## Output files

- Strategy and benchmark CSVs: [data/processed/backtest_results](data/processed/backtest_results)
- Summary CSV: [data/processed/backtest_results/backtest_10pct_summary.csv](data/processed/backtest_results/backtest_10pct_summary.csv)
- Markdown report: [data/processed/backtest_results/backtest_10pct_summary.md](data/processed/backtest_results/backtest_10pct_summary.md)

## Important limitations

- This is a research/backtesting project, not a real trading system.
- The current universe is limited to the 10 configured NSE stocks and is not a complete historical NSE universe.
- Model performance must be judged using out-of-sample evidence, not by assuming a 10% gross target is achievable without losses.

## Research audit (2026-09-28)

Run the probability-threshold and barrier research with:

```powershell
.\.venv\Scripts\python -m src.research_analysis
```

Each run is written to a new timestamped folder under [data/processed/research_results](data/processed/research_results); existing result files are not replaced. The run summarized below is [research_run_20260928T055346Z](data/processed/research_results/research_run_20260928T055346Z).

### Threshold selection

Dates are split chronologically into 60% train, 20% validation, and 20% final test. The probability threshold is selected only on validation rows by F1, from the fixed grid `0.40` to `0.70` in `0.05` increments, requiring at least 10 validation signals. Ties use the first threshold in the grid. The threshold and model configuration are frozen before test metrics are calculated.

| Target | Selected threshold | Validation signals / rows | Test signals / rows | Test target-hit rate / precision | Test recall | Test balanced accuracy | Test confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Target_1D | 0.40 | 206 / 950 | 235 / 960 | 35.32% | 25.46% | 50.74% | `[[482, 152], [243, 83]]` |
| Target_5D | 0.40 | 479 / 950 | 488 / 960 | 39.34% | 52.75% | 51.54% | `[[300, 296], [172, 192]]` |
| Target_30D | 0.45 | 543 / 950 | 472 / 960 | 31.57% | 49.34% | 50.12% | `[[335, 323], [153, 149]]` |

Confusion matrices use `[[true non-UP, predicted UP], [missed UP, correctly predicted UP]]`.

### Barrier model test results

**Correction:** the barrier counts and model metrics in this earlier research-audit section should not be relied on. Its label helper reset each ticker group's index and then assigned labels using those reset indices in the combined dataframe, causing cross-ticker misalignment. The final portfolio evaluator below uses a separate index-preserving barrier-label builder with a multi-ticker regression test; the prior research module and artifacts were left unchanged.

Barrier labels enter at the next trading session Open, use a +10% target and -5% stop, and classify a same-day touch of both barriers as `STOP_HIT`. Each horizon searches the next 1, 3, 5, 7, 10, or 15 trading sessions, including the entry session as day one. The target is `TARGET_HIT` versus `STOP_HIT`/`TIME_EXIT`; test target-hit rate is the fraction of test labels that hit the target.

| Horizon | All labeled classes: target / stop / time | Test classes: target / stop / time | Test N | Target-hit rate / precision | Recall | Balanced accuracy | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 day | 0 / 16 / 1,176 | 0 / 2 / 236 | 238 | 0.00% | 0.00% | 100.00%* | `[[238, 0], [0, 0]]` |
| 3 days | 16 / 130 / 1,046 | 5 / 27 / 206 | 238 | 2.10% | 100.00% | 50.00% | `[[0, 233], [0, 5]]` |
| 5 days | 40 / 260 / 892 | 12 / 61 / 165 | 238 | 5.04% | 100.00% | 50.00% | `[[0, 226], [0, 12]]` |
| 7 days | 62 / 353 / 777 | 20 / 82 / 136 | 238 | 8.40% / 8.44% | 100.00% | 50.23% | `[[1, 217], [0, 20]]` |
| 10 days | 110 / 449 / 633 | 30 / 108 / 100 | 238 | 12.61% / 12.66% | 100.00% | 50.24% | `[[1, 207], [0, 30]]` |
| 15 days | 216 / 568 / 408 | 33 / 140 / 65 | 238 | 13.87% | 100.00% | 50.00% | `[[0, 205], [0, 33]]` |

For barrier confusion matrices, rows are actual non-target/target and columns are predicted non-target/target. `*` The 1-day test slice contains no positive target labels, so its 100% balanced accuracy is only the recall on the sole observed class and is not evidence of target detection. The logistic models for 3–15 days predicted nearly every test sample as a target; their 100% target recall comes with low precision and approximately 50% balanced accuracy. A 1-day majority-class fallback is used because its training labels contain only one class.

### Leakage and interpretation

- The threshold grid, F1 objective, minimum-signal rule, tie-break, and XGBoost configuration are fixed in code; only validation labels determine the selected threshold. Final test labels are used only for reporting. Regression tests mutate final test labels and verify the threshold policy and model configuration do not change.
- Barrier model hyperparameters are fixed before evaluation. Models are fitted on the chronological training split; `StandardScaler` is inside the logistic-regression pipeline and is fitted only by `Pipeline.fit` on training data. Validation and test data are transformed with that fitted scaler.
- There is no learned imputer, feature selector, or probability calibration in this research path. Rows with non-finite or missing features are excluded; no statistics are estimated from validation/test data. Numeric feature columns are selected by dtype and explicit target-column exclusions, not by a fit on the full sample.
- Features are trailing indicators or same-session OHLCV values available after the signal-date close. Future-derived target and barrier columns are excluded from model features. OOF prediction generation uses chronological folds and purges the target horizon plus one date bucket before each fold.
- These barrier labels and metrics do not include transaction costs, slippage, liquidity, or portfolio constraints. Target-hit frequency is not net return, expected profit, or evidence that a trade can reliably achieve 10%. The observed precision and balanced accuracy do not support a profitability claim.

## Final portfolio evaluation (2026-09-28)

Run the evaluator with:

```powershell
.\.venv\Scripts\python -m src.final_portfolio_evaluation
```

It rebuilds technical features in memory from the raw OHLCV files, so the existing processed dataset and prediction files are not rewritten. OOF predictions end on 2026-08-14; the final signal period starts on the next session, 2026-08-17, and the price evaluation ends on 2026-09-25. Signal cutoffs vary by holding period to preserve complete forward price coverage: 2026-09-24, 09-22, 09-18, 09-16, 09-11, and 09-04 for 1, 3, 5, 7, 10, and 15 sessions respectively.

The selected thresholds were frozen from the prior validation result before this post-OOF period: `Target_1D=0.40`, `Target_5D=0.40`, and `Target_30D=0.45`. The original XGBoost configuration was refit on pre-cutoff rows only; the final-period signals are its `Predicted_Label == UP` outputs, compared with those same signals filtered by the frozen threshold. Training rows were purged by target horizon plus one date bucket. Barrier models use fixed logistic-regression settings and a fixed 0.50 positive-probability threshold. No final-period metric was used to choose a threshold, model setting, or strategy parameter.

Portfolio settings were fixed at INR 100,000 starting capital, 20% of current equity per position, at most five concurrent positions, whole-share purchases, no borrowing, 0.10% transaction cost per side, +10% target, and -5% stop. Same-day target/stop ambiguity is resolved stop-first. Remaining positions are liquidated at the final available close. The independent benchmark invests equal amounts in the ten OOF-universe stocks at the first final-period Open and holds through the final close; it does not use model signals.

The final run is [portfolio_run_20260928T060609Z](data/processed/final_portfolio_evaluation/portfolio_run_20260928T060609Z). Full strategy-by-target-by-horizon metrics, trades, daily equity curves, and train/purge metadata are in that folder.

| Strategy family | Portfolio return range | Max drawdown range | Profit factor range | Target hits | Trades across separate runs | Transaction costs across separate runs |
|---|---:|---:|---:|---:|---:|---:|
| Original XGBoost UP | -7.60% to -2.57% | -7.60% to -2.72% | 0.000 to 0.439 | 0% in all runs | 722 | INR 19,562.95 |
| Probability-filtered XGBoost | -6.51% to -2.80% | -6.57% to -2.99% | 0.000 to 0.515 | 0% in all runs | 643 | INR 18,419.45 |
| Barrier models | -4.93% to 0.00% | -5.17% to 0.00% | 0.000 to 0.310 | 0% where trades occurred | 57 | INR 1,976.58 |
| Independent equal-weight buy-and-hold | -7.35% | -7.35% | Not calculated | Not applicable | 10 holdings | INR 181.42 |

Trade counts and costs in the first three rows are sums across separate, non-concurrent counterfactual portfolio runs; they are not one combined portfolio. The 1-day barrier classifier produced no signals/trades.

| Barrier horizon | Test target-hit rate | Trades | Portfolio return | Max drawdown | Profit factor | Transaction costs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 day | N/A (no trades) | 0 | 0.00% | 0.00% | N/A | INR 0.00 |
| 3 days | 0.00% | 14 | -2.22% | -2.22% | 0.310 | INR 505.55 |
| 5 days | 0.00% | 12 | -4.62% | -4.62% | 0.067 | INR 403.20 |
| 7 days | 0.00% | 13 | -4.92% | -4.92% | 0.050 | INR 430.13 |
| 10 days | 0.00% | 11 | -4.93% | -4.93% | 0.111 | INR 387.01 |
| 15 days | 0.00% | 7 | -4.64% | -5.17% | 0.000 | INR 250.70 |

This is a short, 29-session post-OOF market window, with only 15 eligible signal sessions for 15-day holds. All tested strategy variants had zero +10% target hits; the results are highly sample-sensitive and do not establish future performance. This evaluation places no real trades and connects to no brokerage service.

Run a synthetic portfolio check without reading or writing project data:

```powershell
.\.venv\Scripts\python -m src.final_portfolio_evaluation --smoke
```

## Expanding walk-forward portfolio evaluation

Run the longer chronological evaluation with:

```powershell
.\.venv\Scripts\python -m src.walk_forward_portfolio_evaluation
```

The runner rebuilds features from raw OHLCV in memory and excludes all rows after the last OOF prediction date (2026-08-14), including the previous final portfolio holdout. Its original three-window setting used only 60% initial training plus three 8% test windows, so it stopped at the 84% date boundary (2025-09-22) despite later eligible dates. Five non-overlapping expanding test windows fit before the cutoff. Within each window, earlier data is split into purged training and validation segments; two fixed XGBoost candidates (`50` trees/depth `2`, `100` trees/depth `3`) and the probability threshold are selected only on validation data before test signals are generated. Histogram tree building uses one worker for predictable resource use. Barrier models select between fixed logistic `C` values `0.1` and `1.0` on validation data, using a 400-iteration `liblinear` solver. Purge lengths are each target horizon plus one trading-date bucket. Prior final-test metrics are not read or used to tune this evaluation.

The report includes original and probability-filtered XGBoost signals, binary barrier models at all six holding periods, and an independent equal-weight buy-and-hold benchmark. Portfolio settings can be overridden with `--starting-capital`, `--position-size-fraction`, `--max-concurrent-trades`, `--cost-per-side`, `--target-return`, and `--stop-loss`; `--windows` changes the number of test windows. Each run writes to a new timestamped folder under [data/processed/walk_forward_portfolio_evaluation](data/processed/walk_forward_portfolio_evaluation). Outputs include window summaries, ticker contributions, daily equity curves, selected model/threshold settings and split metadata. Return and target-hit uncertainty is estimated with a ticker-cluster bootstrap; with only ten ticker clusters, these intervals are descriptive and do not remove cross-stock dependence or small-window uncertainty.

The original three-window run [portfolio_run_20260928T140344Z](data/processed/walk_forward_portfolio_evaluation/portfolio_run_20260928T140344Z) is preserved. The expanded five-window report is [portfolio_run_20260929T031304Z](data/processed/walk_forward_portfolio_evaluation/portfolio_run_20260929T031304Z). It uses 1,392 eligible dates through 2026-08-14 and leaves August 13–14 unused because test-window sizes are floored to 111 sessions.

| Window | Test dates | Sessions | Equal-weight benchmark return |
|---:|---|---:|---:|
| 1 | 2024-05-23 to 2024-10-30 | 111 | 8.14% |
| 2 | 2024-10-31 to 2025-04-11 | 111 | -8.53% |
| 3 | 2025-04-15 to 2025-09-22 | 111 | 6.39% |
| 4 | 2025-09-23 to 2026-03-04 | 111 | -7.10% |
| 5 | 2026-03-05 to 2026-08-12 | 111 | -3.84% |

The latest date ranges actually used by the nine model/horizon configurations in each window are recorded in `model_selection_and_splits.json`. Selection-fit training end dates range from 2023-09-06 to 2023-10-19 in Window 1, 2024-02-20 to 2024-04-04 in Window 2, 2024-08-06 to 2024-09-17 in Window 3, 2025-01-15 to 2025-02-24 in Window 4, and 2025-06-27 to 2025-08-07 in Window 5. Validation and final pre-test refit end-date ranges are identical within each window: 2024-04-03–2024-05-17, 2024-09-16–2024-10-28, 2025-02-21–2025-04-08, 2025-08-06–2025-09-18, and 2026-01-15–2026-02-27 respectively. The pre-test refit uses training plus validation only, with the required horizon purge; each test starts after those dates.

The window summary CSV contains 210 rows (five windows by 42 strategy/target/horizon combinations); the ticker CSV contains 2,150 rows, including zero-trade ticker rows. Each window has 111 test sessions. Test prediction sample counts range from 930 to 1,100 per model/window, with 96–110 eligible signal sessions depending on holding horizon.

Across separate strategy/target/horizon/window runs in the five-window report, results were:

| Strategy | Mean return | Return range | Mean max drawdown | Profit-factor range | Mean target-hit rate | Trades | Transaction costs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original XGBoost UP | -4.11% | -18.68% to 14.55% | -8.79% | 0.440 to 2.520 | 1.72% | 14,644 | INR 409,500.49 |
| Probability-filtered XGBoost | -3.28% | -17.79% to 11.76% | -7.91% | 0.362 to 3.791 | 3.06% | 10,749 | INR 317,760.03 |
| Barrier models | -1.89% | -12.18% to 9.14% | -5.55% | 0.341 to 3.361 | 7.35% | 1,069 | INR 38,493.68 |

Mean returns are averages across independent strategy configurations, not a compounded portfolio. Trades and costs are summed across those separate runs and must not be interpreted as one executable account. Target-hit rates differ by horizon; a hit is the configured gross +10% price barrier before the -5% stop, not net profitability.

| Barrier horizon | Mean return | Return range | Trades | Mean target-hit rate |
|---:|---:|---:|---:|---:|
| 1 day | 0.00% | 0.00% | 0 | N/A; no signals |
| 3 days | -4.28% | -5.22% to -2.74% | 291 | 1.14% |
| 5 days | -1.95% | -9.25% to 2.67% | 264 | 1.94% |
| 7 days | -3.44% | -12.18% to 0.08% | 212 | 4.70% |
| 10 days | -2.99% | -8.64% to 2.73% | 174 | 9.43% |
| 15 days | 1.31% | -4.28% to 9.14% | 128 | 19.52% |

Each window contains 111 test sessions. Prediction sample counts range from 930 to 1,100 rows per model/window; eligible signal sessions range from 96 to 110 as holding periods reserve complete exits. Ticker-level outcomes for every run are in `window_ticker_summary.csv`. Of 210 strategy rows, 205 have finite 95% ticker-cluster bootstrap intervals; the five 1-day barrier rows have no trades. Intervals resample observed per-ticker net-P&L contributions and do not replay cash constraints or cross-asset dependence, so treat them as descriptive sensitivity estimates, not inferential guarantees.

These results vary materially by window and are based on only three periods and ten tickers. Some individual configurations show positive historical returns, but this is not evidence of durable profitability and does not establish reliable 10% gains. No strategy or threshold was selected using these test outcomes.

Synthetic chronological-split and threshold smoke test:

```powershell
.\.venv\Scripts\python -m src.walk_forward_portfolio_evaluation --smoke
```
