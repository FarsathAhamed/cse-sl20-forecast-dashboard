# Detailed 8-Phase Step-by-Step Execution Guide

> **Companion to:** [`PROJECT_REMAKE_WORKFLOW.md`](PROJECT_REMAKE_WORKFLOW.md)  
> **Target Standard:** [`AIMT_abstract_fernando.pdf`](AIMT_abstract_fernando.pdf) — *Next-day direction and risk forecasting on the Colombo Stock Exchange: a leakage-controlled evaluation of eleven S&P SL20 stocks*  
> **Environment:** Windows PowerShell + Python 3.10+

Follow these 8 phases **strictly in order**. Do not skip a verification gate, and **never inspect hold-out test results before Phase 5–6 rules are frozen**.

---

## Phase 0 (Prerequisite): Workspace Cleanup & Environment Setup

Before starting Phase 1, separate legacy notebooks from the new reproducible codebase so old Yahoo Finance results and leaky notebooks are never accidentally mixed into the remake.

### Step 0.1 — Create the Clean Directory Tree (PowerShell)
```powershell
cd "c:\Users\Bhanuka\Documents\Case 02 project"

New-Item -ItemType Directory -Force -Path `
  "configs", `
  "data\raw_cse", `
  "data\audit", `
  "data\processed", `
  "src", `
  "results\per_stock", `
  "figures", `
  "paper", `
  "archive_old_v1"
```

### Step 0.2 — Archive Legacy Files (Optional but Recommended)
Move old notebooks (`notebooks/`, `models/`, `ww.ipynb`, `exportToHTML/`, old `jkh.csv`) into `archive_old_v1/` so your root directory stays clean. Keep `AIMT_abstract_fernando.pdf` in `paper/` or root.

### Step 0.3 — Create Virtual Environment & Install Pinned Dependencies
Create `requirements.txt`:
```text
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
scipy>=1.13.0
statsmodels>=0.14.2
pyyaml>=6.0.1
matplotlib>=3.9.0
seaborn>=0.13.2
```

Run in PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Gate 0 Check:** `python -c "import pandas, sklearn, statsmodels, scipy, yaml, matplotlib; print('OK')"` prints `OK`.

---

## Phase 1: Pre-Registration & Universe Selection

**Goal:** Lock the study universe, random selection seed, cleaning thresholds, feature list, split boundaries, model grids, and statistical tests *before* touching any model output.

### Step 1.1 — Select the 11 S&P SL20 Stocks Randomly
1. Export the official **S&P Sri Lanka 20 (June 2026)** constituent list.
2. Filter to constituents with **≥ 2,500 trading days** of daily OHLCV data covering **2015-01-01 to 2026-08-31**.
3. Run a deterministic random draw with a fixed seed (e.g., `seed = 42`) to choose **11 stocks**:
```python
import random
eligible_tickers = [
    # Paste all S&P SL20 (June 2026) tickers meeting the >= 2,500 trading days rule
]
rng = random.Random(42)
selected_11 = sorted(rng.sample(eligible_tickers, 11))
print(selected_11)
```

### Step 1.2 — Freeze `configs/protocol.yaml`
Create `configs/protocol.yaml` and **do not edit it after running Phase 6**:

```yaml
study:
  title: "Next-day direction and risk forecasting on the Colombo Stock Exchange"
  universe_source: "S&P Sri Lanka 20 (June 2026)"
  min_trading_days: 2500
  selection_seed: 42
  start_date: "2015-01-01"
  end_date: "2026-08-31"

tickers:
  # Replace with your exact 11 selected ticker filenames (without .csv)
  - "JKH_N0000"
  - "COMB_N0000"
  - "DIAL_N0000"
  - "HAYL_N0000"
  # ... add remaining 7 tickers

cleaning:
  drop_zero_ohlc: true
  drop_identical_prev_row: true
  break_min_calendar_gap_days: 6
  break_min_abs_return: 0.50
  post_break_purge_rows: 30

split:
  train_fraction: 0.80        # Chronological ~Jan 2015 to mid-2024
  embargo_gap_days: 10        # Dropped between train and test
  cv_splits: 5                # TimeSeriesSplit strictly inside training

direction_model:
  random_state: 42
  scoring: "balanced_accuracy"
  rf_param_grid:
    n_estimators: [200, 500]
    max_depth: [3, 5, 8]
    min_samples_leaf: [20, 50]

risk_model:
  random_state: 42
  scoring: "neg_mean_squared_error"
  rf_param_grid:
    n_estimators: [200, 500]
    max_depth: [3, 5, 8]
    min_samples_leaf: [20, 50]

statistics:
  alpha: 0.05
  hac_maxlags: null           # Auto Bartlett/Newey-West bandwidth: floor(4*(T/100)^(2/9))
  multiple_testing: "holm"
```

**Gate 1 Check:** `configs/protocol.yaml` exists, lists exactly 11 tickers, and is committed to version control before any modeling runs.

---

## Phase 2: Data Acquisition & Pre-Modeling Audit

**Goal:** Place all 11 raw exchange-feed CSVs in `data/raw_cse/` and audit every file for data quality issues before writing cleaning logic.

### Step 2.1 — Standardize Raw CSV Placement
Place all 11 files in `data/raw_cse/<TICKER>.csv` with required columns:
`Date, Open, High, Low, Close, Volume` (extra columns like `VWAP` or `Turnover` are fine and ignored if not used).

### Step 2.2 — Build & Run `src/audit_data.py`
Implement `src/audit_data.py` to inspect each CSV and output `data/audit/audit_summary.csv` with the following per-stock diagnostics:
1. `n_rows`, `date_min`, `date_max`
2. `n_zero_ohlc_rows`: count where `(Open <= 0) | (High <= 0) | (Low <= 0) | (Close <= 0)`
3. `n_zero_volume_rows`: count where `Volume <= 0`
4. `n_stale_duplicate_rows`: count where `[Open, High, Low, Close, Volume]` equals the previous row
5. `pct_flat_days`: fraction of days where `Close.shift(-1) == Close`
6. `n_gaps_ge_6d`: count of calendar gaps `(Date - Date.shift(1)).dt.days >= 6`
7. `n_jumps_ge_50pct`: count of `abs(Close.pct_change()) >= 0.50`
8. `min_ret`, `min_ret_date`, `max_ret`, `max_ret_date`

Run:
```powershell
python src/audit_data.py --config configs/protocol.yaml
```

**Gate 2 Check:**
- All 11 stocks pass `n_rows >= 2500`.
- All 11 stocks share the same exchange feed date span (Jan 2015 – Aug 2026).
- `data/audit/audit_summary.csv` is saved and reviewed (confirm known market closures like 2020 COVID and Apr 2022 appear across all stocks, and known splits like JKH 2024 / HAYL 2021 are flagged in `n_jumps_ge_50pct`).

---

## Phase 3: Uniform Cleaning & Structural Break Protocol

**Goal:** Clean every stock identically without manual per-stock hacks, marking structural breaks so no rolling indicator or return ever crosses a split, halt, or market closure.

### Step 3.1 — Implement `src/cleaning.py`
Core deterministic cleaning function (`clean_stock_df(df, cfg)`):

```python
import numpy as np
import pandas as pd

def clean_stock_df(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    n_raw = len(df)

    # 1. Drop all-zero / non-positive OHLC placeholder rows
    valid_price = (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)
    n_zero_dropped = int((~valid_price).sum())
    df = df.loc[valid_price].reset_index(drop=True)

    # 2. Drop exact stale/copied consecutive OHLCV rows
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    is_stale = (df[ohlcv] == df[ohlcv].shift(1)).all(axis=1)
    n_stale_dropped = int(is_stale.sum())
    df = df.loc[~is_stale].reset_index(drop=True)

    # 3. Detect structural breaks (calendar gap >= 6 days OR |return| >= 50%)
    cal_gap = df["Date"].diff().dt.days
    raw_ret = df["Close"].pct_change()
    is_break = (
        (cal_gap >= cfg["cleaning"]["break_min_calendar_gap_days"]) |
        (raw_ret.abs() >= cfg["cleaning"]["break_min_abs_return"])
    ).fillna(False)

    df["is_break"] = is_break
    df["regime_id"] = is_break.cumsum()

    # 4. Flag post-break purge rows (break day + next 30 rows within each regime)
    purge_n = cfg["cleaning"]["post_break_purge_rows"]
    df["days_since_break"] = df.groupby("regime_id").cumcount()
    # Regime 0 has no preceding break (only natural rolling warmup); regimes >= 1 purge 30 rows
    df["in_purge_window"] = (df["regime_id"] > 0) & (df["days_since_break"] <= purge_n)

    meta = {
        "n_raw": n_raw,
        "n_zero_dropped": n_zero_dropped,
        "n_stale_dropped": n_stale_dropped,
        "n_breaks": int(is_break.sum()),
        "break_dates": df.loc[is_break, "Date"].dt.strftime("%Y-%m-%d").tolist(),
        "n_purged_rows": int(df["in_purge_window"].sum()),
    }
    return df, meta
```

### Step 3.2 — Verify Break Detection
Run a quick check across all 11 stocks and inspect `meta["break_dates"]` and `meta["n_purged_rows"]`.

**Gate 3 Check:**
- Zero rows with `Close <= 0`, `High < Low`, or `NaN` dates remain.
- Every major split (e.g., -90% drop) and market closure gap (≥ 6 days) increments `regime_id`.
- Purged rows are ~3–8% of each series (not unexpectedly large).

---

## Phase 4: Stationary Feature & Target Engineering

**Goal:** Construct strictly causal (day $t$ or earlier) stationary features and next-day ($t+1$) targets **inside each continuous `regime_id`** so no rolling window or `shift(-1)` target ever crosses a structural break.

### Step 4.1 — Implement `src/features.py`
Compute all indicators **grouped by `regime_id`**:

1. **12 Stationary Direction Features:**
   - `ret1` = $\ln(\text{Close}_t / \text{Close}_{t-1})$
   - `ret_lag2`, `ret_lag3`, `ret_lag5` = `ret1.shift(1)`, `ret1.shift(2)`, `ret1.shift(4)`
   - `roc10` = $\text{Close}_t / \text{Close}_{t-10} - 1$
   - `ma10_rel` = $\text{Close}_t / \text{MA}_{10}(\text{Close}) - 1$
   - `ma30_rel` = $\text{Close}_t / \text{MA}_{30}(\text{Close}) - 1$
   - `rsi` = 14-day RSI using rolling gains and losses
   - `macd_rel` = $(\text{EMA}_{12} - \text{EMA}_{26}) / \text{Close}_t$
   - `stoch_k` = $100 \times (\text{Close}_t - \min_{14}(\text{Low})) / (\max_{14}(\text{High}) - \min_{14}(\text{Low}))$
   - `willr` = $-100 \times (\max_{14}(\text{High}) - \text{Close}_t) / (\max_{14}(\text{High}) - \min_{14}(\text{Low}))$
   - `vol_rel` = $\ln(\text{Volume}_t / \text{MA}_{20}(\text{Volume}_t))$ (safely replace 0 with `NaN` before `log`)

2. **HAR Risk Features:**
   - Daily price range proxy: `hl_range` = $(\text{High}_t - \text{Low}_t) / \text{Close}_t$
   - `har_d` = `hl_range` (1-day)
   - `har_w` = `hl_range.rolling(5).mean()` (5-day)
   - `har_m` = `hl_range.rolling(22).mean()` (22-day)

3. **Targets (computed strictly within `regime_id` so day $t$ before a break has `NaN` target):**
   - `y_dir` = `(Close.shift(-1) > Close).astype(float)` where `Close.shift(-1).notna()`
   - `y_risk` = `hl_range.shift(-1)`
   - *(Optional sensitivity target)* `y_absret` = `ret1.shift(-1).abs()`

4. **Filter Valid Modeling Rows:**
   - Drop rows where `in_purge_window == True`.
   - Replace `±inf` with `NaN`.
   - Build two clean task views:
     - `df_dir`: `dropna(subset=DIR_FEATURES + ["y_dir"])`
     - `df_risk`: `dropna(subset=HAR_FEATURES + ["y_risk"])` (and optionally `DIR_FEATURES` if RF risk uses both HAR + technical indicators — keep consistent with `configs/protocol.yaml`).
   - Save processed files to `data/processed/<TICKER>_features.parquet` (or `.csv`).

**Gate 4 Check (Leakage Unit Test):**
- Verify `df_dir["Date"].is_monotonic_increasing == True`.
- Verify **no** target `y_dir` or `y_risk` is computed on the last row before any `is_break == True` date.
- Verify no raw price (`Open`, `High`, `Low`, `Close`) or raw `Volume` column is in `DIR_FEATURES`.

---

## Phase 5: Leakage-Controlled Chronological Split

**Goal:** Partition each stock strictly by time with a 10-day embargo gap between training and hold-out test sets.

### Step 5.1 — Split Function (Shared Across Direction & Risk)
```python
def chronological_split(df: pd.DataFrame, train_frac: float = 0.80, gap: int = 10):
    n = len(df)
    cut = int(n * train_frac)
    train = df.iloc[: cut - gap].copy()
    test = df.iloc[cut:].copy()
    assert train["Date"].max() < test["Date"].min(), "Chronological leakage detected!"
    return train, test
```

**Gate 5 Check:**
- Training spans roughly **Jan 2015 – mid-2024**.
- Hold-out test spans roughly **mid-2024 – Aug 2026**, with **488–523 days** for direction (matching [`AIMT_abstract_fernando.pdf`](AIMT_abstract_fernando.pdf)).
- Strictly `train["Date"].max() < test["Date"].min()` with at least 10 trading days removed in between.

---

## Phase 6: Modeling & Benchmarks (Direction & Risk)

**Goal:** Fit and tune models strictly on `train`, then evaluate **once** on `test`.

### Step 6.1 — Direction Track (`src/models_direction.py`)
1. **Random Forest Classifier:**
   - `GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=TimeSeriesSplit(5), scoring="balanced_accuracy")` fit on `(X_train, y_train)`.
   - Extract `best_estimator_`, predict `y_pred_rf` on `X_test`.
2. **Benchmarks (evaluated on exact same `y_test`):**
   - **Majority-Class Rule:** `major_class = int(y_train.mean() >= 0.5)`; `y_pred_major = np.full(len(y_test), major_class)`. *(Also record test majority rate `max(y_test.mean(), 1 - y_test.mean())` for exact binomial comparison.)*
   - **Previous-Day Direction:** `y_pred_prev = (test["ret1"] > 0).astype(int)`. *(Uses `ret1` from day $t$, which is already known at close of day $t$, to predict day $t+1$ — zero leakage.)*
   - **Logistic Regression:** `Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])` fit on `(X_train, y_train)`, predict `y_pred_lr`.
3. **Permutation Importance + Noise Benchmark:**
   - Compute `permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=42, scoring="balanced_accuracy")`.
   - *(Optional diagnostic)* Add 3 Gaussian noise columns to confirm whether real technical indicators outperform pure noise on `X_test`.

### Step 6.2 — Risk / Volatility Track (`src/models_risk.py`)
1. **HAR Model (OLS):**
   - Fit `LinearRegression()` on `train[["har_d", "har_w", "har_m"]]` -> `train["y_risk"]`.
   - Predict `y_pred_har` on `test[["har_d", "har_w", "har_m"]]`.
2. **Random Forest Regressor:**
   - `GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=TimeSeriesSplit(5), scoring="neg_mean_squared_error")` fit on training features.
   - Predict `y_pred_rf_risk` on test features.
3. **Benchmarks:**
   - **Historical Average:** `y_pred_hist = np.full(len(test), train["y_risk"].mean())`.
   - **22-Day Rolling Mean:** `y_pred_m22 = test["har_m"].values`.

**Gate 6 Check:**
- All scalers, means, and CV grids are fit on `train` only.
- Record total trial count per stock (e.g., 18 RF direction fits + 18 RF risk fits + LogReg + HAR + baselines).

---

## Phase 7: Statistical Testing & Multi-Stock Aggregation

**Goal:** Compute rigorous out-of-sample metrics, HAC-robust Diebold–Mariano tests, and Holm–Bonferroni multiple-testing adjustments across all 11 stocks.

### Step 7.1 — Implement `src/stats_tests.py`
Include four clean statistical utilities:

1. **Wilson 95% Confidence Interval** for direction accuracy (`statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`).
2. **Exact One-Sided Binomial Test** for H1 (`scipy.stats.binomtest(k_correct, n_test, p=p_majority, alternative="greater")`).
3. **Out-of-Sample $R^2$:**
   $$R^2_{\text{OOS}} = 1 - \frac{\sum (y_t - \hat{y}_{t,\text{model}})^2}{\sum (y_t - \hat{y}_{t,\text{bench}})^2}$$
4. **One-Sided Diebold–Mariano Test with Newey–West (HAC) Robust Errors:**
   ```python
   import numpy as np
   import statsmodels.api as sm
   from scipy.stats import norm

   def dm_test_hac(y_true, y_pred_model, y_pred_bench, maxlags=None):
       e_m = np.asarray(y_true) - np.asarray(y_pred_model)
       e_b = np.asarray(y_true) - np.asarray(y_pred_bench)
       # Positive d_t means model has LOWER squared error than benchmark
       d = e_b**2 - e_m**2
       n = len(d)
       if maxlags is None:
           maxlags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
       ols = sm.OLS(d, np.ones((n, 1))).fit(
           cov_type="HAC", cov_kwds={"maxlags": maxlags}
       )
       t_stat = float(ols.tvalues[0])
       p_one_sided = float(1.0 - norm.cdf(t_stat))
       return {"dm_stat": t_stat, "p_raw_one_sided": p_one_sided, "maxlags": maxlags}
   ```
5. **Holm–Bonferroni Adjustment Across 11 Stocks:**
   ```python
   from statsmodels.stats.multitest import multipletests
   _, p_holm, _, _ = multipletests(raw_p_values, alpha=0.05, method="holm")
   ```

### Step 7.2 — Run Master Pipeline & Aggregator
```powershell
python src/run_all_stocks.py --config configs/protocol.yaml
python src/summarize_results.py --results-dir results/per_stock --out results/summary_11_stocks.csv
```

**Gate 7 Check:**
- Exactly 11 JSON files exist in `results/per_stock/`.
- `results/summary_11_stocks.csv` contains one row per stock with:
  - `ticker`, `n_train`, `n_test_dir`, `n_test_risk`
  - `rf_acc`, `rf_acc_ci_low`, `rf_acc_ci_high`, `rf_bal_acc`, `rf_mcc`, `maj_acc`, `prev_acc`, `lr_acc`, `dir_p_raw`, `dir_p_holm`
  - `har_r2_vs_hist`, `har_dm_hist_p_raw`, `har_dm_hist_p_holm`, `har_r2_vs_m22`, `har_dm_m22_p_raw`, `har_dm_m22_p_holm`, `rf_risk_r2_vs_hist`, `har_vs_rf_p_holm`

---

## Phase 8: Freeze Results, Generate Figures & Manuscript Verification

**Goal:** Lock all numerical outputs, produce publication-quality figures programmatically from frozen JSONs, and verify every number in your abstract/paper.

### Step 8.1 — Generate Publication Figures (`src/make_figures.py`)
```powershell
python src/make_figures.py --summary results/summary_11_stocks.csv --out-dir figures
```
Generate 4 clean figures (using **LKR** currency labels, 300 DPI, clear legends):
1. `figures/fig1_direction_vs_baselines.png` — Grouped bar/dot plot of RF accuracy (with 95% Wilson CI) vs. Majority Class, Previous-Day, and Logistic Regression across all 11 stocks.
2. `figures/fig2_risk_har_oos_r2.png` — Horizontal bar plot of HAR vs. RF out-of-sample $R^2$ across all 11 stocks, marking Holm-significant stocks (`*` for adjusted $p < 0.05$).
3. `figures/fig3_price_series_breaks.png` — Representative stock price trajectories (in LKR) with detected structural breaks (splits, halts, 2020/2022 closures) and the train/test cutoff shaded.
4. `figures/fig4_permutation_importance.png` — Hold-out permutation importance summary showing that technical indicators cluster near zero/noise for direction.

### Step 8.2 — Freeze & Verify Against Abstract / Full Paper
Complete `paper/manuscript_checklist.md` before submission:
- [ ] Every percentage, sample size (`488–523`), $R^2$ range (`12–42%`), and significance count (`0/11` direction, `7/11` HAR vs. historical mean, `5/11` HAR vs. 22-day mean) matches `results/summary_11_stocks.csv` exactly.
- [ ] All currency labels use **LKR** (no `$` symbols).
- [ ] All ticker symbols and date ranges (`Jan 2015 – mid-2024` train, `mid-2024 – Aug 2026` test) are consistent across abstract, methods, tables, and charts.
- [ ] Limitations explicitly note thin trading, high flat-day frequencies, structural breaks/splits, shared 2024–2026 market rally, and non-tradability without transaction costs.
- [ ] Citations are verified (e.g., Corsi 2009, Diebold & Mariano 1995, Newey & West 1987, Bailey et al. 2014/2016, Kumbure et al. 2022) and AI assistance is disclosed if required by the conference.

---

## Quick Command Summary (Full Run Order)

```powershell
# 0. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Audit raw 11-stock CSVs
python src/audit_data.py --config configs/protocol.yaml

# 3–7. Run cleaning, features, chronological split, direction & risk models, and tests
python src/run_all_stocks.py --config configs/protocol.yaml

# 7. Aggregate & apply Holm-Bonferroni correction across all 11 stocks
python src/summarize_results.py --results-dir results/per_stock --out results/summary_11_stocks.csv

# 8. Generate publication figures from frozen results
python src/make_figures.py --summary results/summary_11_stocks.csv --out-dir figures
```
