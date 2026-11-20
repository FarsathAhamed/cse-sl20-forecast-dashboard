# CSE Stock Forecast Dashboard — 4-Step Build Guide & Prompt Playbook

> **App Goal:** Build a portfolio-ready **Streamlit** web application for your leakage-controlled Colombo Stock Exchange (S&P SL20) forecasting project. Users can pick any of the **11 S&P SL20 stocks** and view the **next-day direction (`UP` / `NOT-UP`) prediction**, **confidence score**, **next-day risk/volatility forecast**, and an **interactive LKR price chart**, backed by your frozen research results.
>
> **Core Rule:** Do **not** modify frozen research files (`configs/protocol.yaml`, `data/processed/`, `results/summary_11_stocks.csv`, or `results/per_stock/*.json`). The dashboard reads them strictly in read-only mode.
>
> **How to use:** Copy and paste **one step prompt at a time** into the chat. Complete the **Gate Check** for each step before moving to the next one.

---

## Target App Architecture

```text
Case 02 project/
├── configs/protocol.yaml            # Existing frozen protocol (11 tickers, features)
├── data/raw_cse/                    # Existing raw OHLCV data (for price charts)
├── data/processed/                  # Existing stationary features (*_dir.csv, *_risk.csv)
├── results/
│   ├── summary_11_stocks.csv        # Existing 11-stock benchmark & Holm-adjusted p-values
│   └── per_stock/*.json             # Existing per-stock metrics, best_params, HAR coefs
├── artifacts/models/                # [Step 1] Cached/saved direction & risk models (.joblib)
├── app/
│   ├── predictor.py                 # [Step 1] Data loader + leakage-safe inference engine
│   ├── charts.py                    # [Step 2 & 3] Interactive Plotly charts
│   └── streamlit_app.py             # [Step 2, 3 & 4] Main Streamlit dashboard
├── .streamlit/config.toml           # [Step 4] Clean dark/finance theme config
└── README.md                        # [Step 4] GitHub portfolio README + live demo badge
```

---

## Step 1 — Environment Setup, Model Persistence & Inference Engine

### What this step builds
- Installs dashboard packages (`streamlit`, `plotly`, `joblib`) and updates `requirements.txt` for Streamlit Cloud compatibility.
- Creates `app/predictor.py` to:
  1. Load the 11 S&P SL20 stocks and feature lists from `configs/protocol.yaml`.
  2. Fit and cache (in `artifacts/models/`) each stock's **Direction RandomForest** (using the frozen `best_params` from `results/per_stock/<SYMBOL>.json`, strictly on the leakage-controlled training split) and **HAR Risk model** (using the exact training split / frozen coefficients).
  3. Return a clean prediction dictionary for any selected ticker:
     - `latest_date`, `latest_close_lkr`, `daily_change_pct`
     - `direction_label` (`"UP"` or `"NOT-UP"`), `prob_up`, `confidence_score` (`max(prob_up, 1 - prob_up)`)
     - `predicted_risk_range_pct` (next-day expected `(High - Low) / Close` from HAR) and `risk_level` (`Low` / `Medium` / `High` relative to the stock's historical range distribution)
     - Hold-out benchmark metadata from `results/per_stock/<SYMBOL>.json`.

### Copy-Paste Prompt (Step 1)
```text
Follow Step 1 of DASHBOARD_4STEP_GUIDE.md to build the backend inference engine for the CSE Stock Forecast Dashboard.

1. Add streamlit, plotly, and joblib to requirements.txt and install them into the active .venv environment.
2. Create app/predictor.py (and app/__init__.py) that reads configs/protocol.yaml, data/raw_cse/, data/processed/, results/per_stock/*.json, and results/summary_11_stocks.csv in read-only mode (do NOT modify any existing research files).
3. Implement clean helper functions:
   - get_stock_catalog(): returns the 11 tickers and company names from configs/protocol.yaml.
   - load_price_history(symbol): loads and cleans OHLCV price history from data/raw_cse/<file_stem>.csv for charting.
   - train_or_load_models(symbol): fits the Direction RandomForestClassifier (using frozen best_params from results/per_stock/<symbol>.json on the exact chronological training split from src/split.py) and HAR LinearRegression risk model, saving them to artifacts/models/ with joblib so reloads are instant.
   - get_latest_forecast(symbol): computes the next-day direction prediction (UP / NOT-UP), probability of UP, confidence score (max(p_up, 1 - p_up)), HAR next-day intraday volatility/range forecast (%), risk tier (Low / Medium / High based on training quantiles), and loads frozen hold-out metrics from results/per_stock/<symbol>.json.
4. Run a quick CLI smoke test on JKH and COMB to verify predictions, confidence scores, and risk outputs print cleanly with zero errors.

At the end, explain briefly what was built and confirm the Step 1 Gate Check passed.
```

### Gate Check (Step 1)
- `artifacts/models/` contains saved `.joblib` files for tested tickers.
- Running a quick Python test for `JKH` and `COMB` prints valid `UP`/`NOT-UP`, a confidence score between `50%` and `100%`, and a positive HAR risk range %.

---

## Step 2 — Core Streamlit Dashboard & Interactive Price Chart

### What this step builds
- Creates `app/charts.py` and `app/streamlit_app.py`.
- Sidebar controls:
  - Stock selector dropdown (`JKH — John Keells Holdings PLC`, etc., across all 11 S&P SL20 stocks).
  - Price chart timeframe selector (`3M`, `6M`, `1Y`, `3Y`, `All`) and chart style toggle (`Candlestick` / `Close Line` with optional 10-day & 30-day moving averages).
- Top KPI row (4 metric cards):
  1. **Latest Close (LKR)** + daily % change
  2. **Next-Day Direction** (`UP` / `NOT-UP`)
  3. **Model Confidence** (`XX.X%` + probability bar for `P(Up)` vs `P(Not-Up)`)
  4. **Expected Next-Day Risk (HAR)** (`X.XX%` intraday range + `Low / Medium / High` badge)
- Interactive **Plotly Price & Volume Chart** (in LKR) with hover tooltips, moving averages, and clean layout.
- Honest **Research Transparency Banner** explaining that this is a leakage-controlled academic study (not financial advice) where volatility/risk forecasts beat benchmarks while daily direction remains close to baseline.

### Copy-Paste Prompt (Step 2)
```text
Follow Step 2 of DASHBOARD_4STEP_GUIDE.md to build the core Streamlit dashboard and interactive price chart.

1. Create app/charts.py with reusable Plotly chart builders:
   - build_price_chart(df_price, symbol, chart_type="Candlestick", show_ma=True): interactive OHLC candlestick or close-line chart in LKR with optional 10-day and 30-day moving averages and a volume subplot below.
   - build_confidence_gauge(prob_up, direction_label): clean horizontal probability/confidence visual showing P(UP) vs P(NOT-UP).
2. Create app/streamlit_app.py using @st.cache_data and @st.cache_resource for fast UI updates:
   - Wide layout with a clean header ("CSE S&P SL20 Forecast & Risk Dashboard").
   - Sidebar with stock selector (all 11 stocks from configs/protocol.yaml), date-window selector (3M, 6M, 1Y, 3Y, All), chart type (Candlestick / Line), and MA overlay toggle.
   - 4 KPI cards at the top: Latest Close (LKR), Next-Day Direction (UP / NOT-UP), Confidence Score (%), and Expected Next-Day Risk Range (%) with risk tier.
   - Main interactive price & volume chart and confidence visual.
   - Clear info banner noting this is a leakage-controlled research prototype (not financial advice) and summarizing the stock's hold-out context.
3. Verify the app launches cleanly with `streamlit run app/streamlit_app.py` (no runtime exceptions or missing imports).

At the end, explain how to run the dashboard locally and confirm the Step 2 Gate Check passed.
```

### Gate Check (Step 2)
- `streamlit run app/streamlit_app.py` opens in the browser with no tracebacks.
- Switching between `JKH`, `COMB`, `DIAL`, and `LOLC` updates the 4 KPI cards, confidence bar, and Plotly price chart cleanly.

---

## Step 3 — Model Evidence, Feature Drivers & All-11-Stocks Explorer

### What this step builds
Turns the app from a basic demo into a **high-value Data Science portfolio showcase** by organizing the page into 3 clean tabs:
1. **Tab 1 — Live Forecast & Price Chart** (from Step 2).
2. **Tab 2 — Model Explainability & Backtest Evidence** (for the selected stock):
   - Bar chart of **Hold-Out Permutation Feature Importance** (`permutation_importance` from `results/per_stock/<SYMBOL>.json`) showing which stationary features drive the model.
   - **Benchmark Comparison Table & Chart**: Random Forest vs. Majority Class, Previous-Day Rule, and Logistic Regression (plus Wilson 95% CI and Holm-adjusted p-value), and HAR vs. Historical Mean / 22-Day Mean (`OOS R²` & Diebold–Mariano Holm p-value).
   - **Confusion Matrix** heatmap from the frozen hold-out evaluation.
3. **Tab 3 — S&P SL20 Market Overview (All 11 Stocks)**:
   - Interactive sortable table built from `results/summary_11_stocks.csv` + latest forecast signal for all 11 stocks.
   - Cross-stock comparison chart showing Direction Accuracy vs. Majority Baseline and HAR Risk `OOS R²` across all 11 stocks.

### Copy-Paste Prompt (Step 3)
```text
Follow Step 3 of DASHBOARD_4STEP_GUIDE.md to add explainability, benchmark evidence, and the 11-stock S&P SL20 overview tabs.

1. Extend app/charts.py with Plotly visuals built strictly from frozen results/per_stock/<symbol>.json and results/summary_11_stocks.csv:
   - build_feature_importance_chart(perm_importance_dict): sorted horizontal bar chart of hold-out permutation importance (with error bars for std).
   - build_confusion_matrix_chart(cm, labels): clean annotated Plotly heatmap.
   - build_all_stocks_comparison_chart(summary_df): two-panel or toggleable comparison of (a) RF Direction Accuracy vs Majority Baseline and (b) HAR Risk Out-of-Sample R^2 across all 11 stocks, highlighting statistically significant Holm-adjusted results.
2. Refactor app/streamlit_app.py into 3 clean tabs:
   - Tab 1: "Forecast & Price Chart" (KPI cards, confidence visual, interactive price/volume chart, methodology note).
   - Tab 2: "Model Insights & Validation" (selected stock's hold-out metrics vs. baselines, permutation feature importance chart, confusion matrix, and plain-English interpretation of what the statistical tests mean).
   - Tab 3: "S&P SL20 Overview (11 Stocks)" (summary table across all 11 stocks + cross-stock benchmark comparison chart + CSV download button).
3. Test all 3 tabs across multiple stocks to confirm zero errors, missing keys, or formatting bugs.

At the end, summarize what recruiters/reviewers can inspect in each tab and confirm the Step 3 Gate Check passed.
```

### Gate Check (Step 3)
- All 3 tabs render cleanly for every ticker in the dropdown.
- Numbers shown in Tab 2 & Tab 3 match `results/summary_11_stocks.csv` and `results/per_stock/<SYMBOL>.json` exactly.

---

## Step 4 — UI Polish, Tests, GitHub README & Streamlit Cloud Deployment

### What this step builds
- `.streamlit/config.toml` for a crisp, cohesive financial dashboard theme.
- Lightweight `tests/test_app_smoke.py` (`pytest` or `unittest`) verifying data loading, model inference, and JSON consistency across all 11 stocks.
- Pre-generates all 11 cached models (or confirms fast first-run startup on Streamlit Community Cloud).
- Creates a recruiter-ready `README.md` at the project root featuring:
  - Project headline, badges (Python, Streamlit, scikit-learn, Leakage-Controlled), and live demo link placeholder.
  - Key findings summary (0/11 direction beating majority after Holm correction vs. strong HAR volatility predictability across S&P SL20 stocks — showing scientific honesty).
  - Clean architecture diagram, screenshots section, and 3-command local run instructions.
- Provides a short **LinkedIn post draft** and **Streamlit Community Cloud deployment steps**.

### Copy-Paste Prompt (Step 4)
```text
Follow Step 4 of DASHBOARD_4STEP_GUIDE.md to polish, test, and package the dashboard for GitHub, Streamlit Cloud, and LinkedIn.

1. Create .streamlit/config.toml with a clean, readable financial dashboard theme and polish spacing, number formatting (LKR, %, p-values), and tooltips in app/streamlit_app.py.
2. Create tests/test_app_smoke.py that validates:
   - All 11 tickers load cleanly from configs/protocol.yaml and data/raw_cse/.
   - get_latest_forecast(symbol) returns valid direction, confidence in [0.5, 1.0], and positive risk forecast for all 11 stocks.
   - All frozen JSON metrics load without missing keys.
   Run the smoke test suite and confirm 100% pass.
3. Verify requirements.txt has clean, minimal dependencies compatible with Streamlit Community Cloud (and check .gitignore so .venv/, .idea/, __pycache__/, and temporary files are excluded).
4. Create a portfolio-grade README.md at the project root (concise, visual, recruiter-friendly) covering: project summary, live dashboard features, leakage-controlled methodology, key empirical findings, folder structure, how to run locally, and placeholders for dashboard screenshots + live Streamlit Cloud URL.
5. In your final response, also include:
   - Exact steps to deploy app/streamlit_app.py on Streamlit Community Cloud.
   - A ready-to-post LinkedIn showcase summary (short, clear, with relevant skills/tags).

Confirm the Step 4 Gate Check passed.
```

### Gate Check (Step 4)
- `python -m unittest tests/test_app_smoke.py` passes for all 11 stocks.
- `.gitignore` excludes `.venv/`, `.idea/`, and `__pycache__/`.
- `README.md` is ready for GitHub and `app/streamlit_app.py` runs cleanly from the repo root.

---

## Quick Portfolio Checklist (After Step 4)

- [ ] Push clean repo to GitHub (double-check no `.venv/` or legacy clutter is pushed).
- [ ] Deploy `app/streamlit_app.py` for free on **Streamlit Community Cloud** and paste the live URL into `README.md` and your GitHub repo "About" section.
- [ ] Record a 20–30 second screen recording (GIF/MP4) switching between stocks and tabs (`Forecast` -> `Model Insights` -> `S&P SL20 Overview`).
- [ ] Publish on LinkedIn with the demo video/GIF, GitHub link, live app link, and 3 concise takeaways.
