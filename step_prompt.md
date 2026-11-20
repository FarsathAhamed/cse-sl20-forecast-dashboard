# Step-by-Step AI Prompt Templates (Phases 0–8)

> **Companion files:** [`PROJECT_REMAKE_WORKFLOW.md`](PROJECT_REMAKE_WORKFLOW.md) | [`EXECUTION_GUIDE_8_PHASES.md`](EXECUTION_GUIDE_8_PHASES.md)  
> **How to use:** Copy and paste **one phase prompt at a time**. Read the AI's **Learning & Critical Thinking Summary** at the end of each phase and verify the Gate Check before moving to the next phase.

---

## Standard Learning Footer (Included in Every Phase)
Every prompt below instructs the AI to end with:
1. **What we just did (Plain English):** Simple explanation of the output and why it matters.
2. **How this research step works:** The core data science / finance concept behind it.
3. **Critical thinking for next time:** Common mistakes to avoid, what a conference reviewer could challenge, and what to verify before moving to the next phase.

---

## Phase 0 — Workspace Cleanup & Environment Setup

```text
Follow Phase 0 of EXECUTION_GUIDE_8_PHASES.md.
1. Create the clean folder structure (configs, data/raw_cse, data/audit, data/processed, src, results/per_stock, figures, paper, archive_old_v1).
2. Create requirements.txt with pinned core packages and verify the Python environment.
3. Do NOT delete any old files — only organize/archive legacy notebooks safely if needed.

At the end, explain in simple terms:
- What we did and why separating old vs. new code matters for reproducible research.
- How project structure prevents accidental number/file mix-ups.
- Critical thinking: What should I check before starting Phase 1?
```

---

## Phase 1 — Pre-Registration & Universe Selection

```text
Follow Phase 1 of EXECUTION_GUIDE_8_PHASES.md.
Here is my list of eligible S&P SL20 (June 2026) stocks with >= 2,500 trading days: [PASTE TICKER LIST].
1. Randomly select 11 tickers using a fixed seed (seed = 42).
2. Create and freeze configs/protocol.yaml with all rules (dates, cleaning thresholds, 80/20 split + 10-day gap, CV grids, HAC DM test, Holm correction).
3. Do NOT run any models yet.

At the end, explain in simple terms:
- What pre-registration and random stock selection are, and why we lock configs/protocol.yaml before seeing results.
- How this prevents cherry-picking and p-hacking.
- Critical thinking: What could go wrong if I change a rule or drop a stock after seeing bad accuracy later?
```

---

## Phase 2 — Data Acquisition & Pre-Modeling Audit

```text
Follow Phase 2 of EXECUTION_GUIDE_8_PHASES.md.
My 11 raw stock CSV files are in data/raw_cse/.
1. Build and run src/audit_data.py using configs/protocol.yaml.
2. Save data/audit/audit_summary.csv showing rows, date range, zero-price rows, zero-volume rows, stale/duplicate rows, flat-day %, gaps >= 6 days, and jumps >= 50% for every stock.
3. Show me the full audit table. Do NOT build models yet.

At the end, explain in simple terms:
- What data problems we found across the 11 CSE stocks (splits, halts, closures, flat days).
- How free/exchange frontier-market data differs from textbook US stock data.
- Critical thinking: Which stock looks riskiest or thinnest, and what should I double-check before cleaning?
```

---

## Phase 3 — Uniform Cleaning & Structural Break Protocol

```text
Follow Phase 3 of EXECUTION_GUIDE_8_PHASES.md.
1. Build src/cleaning.py implementing clean_stock_df() using the exact thresholds in configs/protocol.yaml (drop zero OHLC, drop stale consecutive duplicates, flag breaks for gaps >= 6 days or |return| >= 50%, assign regime_id, and flag 30-row post-break purge windows).
2. Run a verification check across all 11 stocks and report how many rows and breaks were flagged per stock.

At the end, explain in simple terms:
- What structural breaks are and why we purge 30 rows after a split or market closure instead of guessing split ratios or keeping fake -90% returns.
- How regime_id protects rolling indicators from crossing breaks.
- Critical thinking: Why is using one uniform rule for all 11 stocks better than manually fixing each stock differently?
```

---

## Phase 4 — Stationary Feature & Target Engineering

```text
Follow Phase 4 of EXECUTION_GUIDE_8_PHASES.md.
1. Build src/features.py to compute all 12 stationary direction features, the 3 HAR range features (daily, 5-day, 22-day), and next-day targets (y_dir and y_risk) strictly inside each regime_id.
2. Exclude purge-window rows, remove inf/NaN rows, and save processed feature files to data/processed/.
3. Run leakage unit tests confirming monotonic dates, no raw price/volume levels in features, and no target crossing a break.

At the end, explain in simple terms:
- Why raw prices (Open/Close/OBV) fail in machine learning and how stationary returns/ratios fix this.
- What the difference is between predicting Direction (H1) vs. Risk/Volatility (H2), and why we keep flat days in the primary analysis.
- Critical thinking: How can a tiny shift(-1) mistake silently create fake 65%+ accuracy (look-ahead leakage), and how did we verify it didn't happen here?
```

---

## Phase 5 — Leakage-Controlled Chronological Split

```text
Follow Phase 5 of EXECUTION_GUIDE_8_PHASES.md.
1. Implement the chronological 80/20 split with a 10-day embargo gap between training (Jan 2015 – mid-2024) and hold-out test (mid-2024 – Aug 2026).
2. Print a summary table of train/test date ranges and sample sizes (n_train, n_test_dir, n_test_risk) for all 11 stocks.
3. Assert strictly that train_max_date < test_min_date with a 10-trading-day gap.

At the end, explain in simple terms:
- Why random train/test shuffling leaks future information in time-series data (and why our old project got 93% train accuracy).
- Why we add a 10-day embargo gap between training and testing.
- Critical thinking: What happens to our evaluation if the 2024–2026 test window is a strong market rally, and how should we discuss that in our limitations?
```

---

## Phase 6 — Modeling & Benchmarks (Direction & Risk)

```text
Follow Phase 6 of EXECUTION_GUIDE_8_PHASES.md.
1. Build src/models_direction.py: tune RandomForestClassifier with TimeSeriesSplit(5) strictly on training data, evaluate once on hold-out, and compare against Majority Class, Previous-Day Direction, and Logistic Regression. Also compute hold-out permutation importance.
2. Build src/models_risk.py: fit HAR (OLS) and tuned RandomForestRegressor strictly on training data, evaluate once on hold-out, and compare against Historical Average and 22-Day Rolling Mean.
3. Record total CV/model trials run per stock.

At the end, explain in simple terms:
- Why accuracy is meaningless without naive benchmarks (like Majority Class or 22-Day Mean).
- Why a simple 3-feature linear HAR model often beats a complex Random Forest on daily financial risk.
- Critical thinking: If someone asks "Why didn't you use LSTM, CNN, or Transformer?", how should I answer based on fair evaluation and trial-count discipline?
```

---

## Phase 7 — Statistical Testing & Multi-Stock Aggregation

```text
Follow Phase 7 of EXECUTION_GUIDE_8_PHASES.md.
1. Build src/stats_tests.py (Wilson 95% CI, exact one-sided binomial test, out-of-sample R^2, one-sided Diebold–Mariano test with Newey–West HAC robust errors, and Holm–Bonferroni multiple-testing adjustment).
2. Build and run src/run_all_stocks.py and src/summarize_results.py to save frozen JSONs in results/per_stock/ and the master table in results/summary_11_stocks.csv.
3. Show me the final 11-stock summary table with both raw and Holm-adjusted p-values.

At the end, explain in simple terms:
- What HAC (Newey–West) robust errors and Diebold–Mariano tests actually test in plain words.
- Why we MUST apply Holm–Bonferroni correction when testing 11 stocks at once.
- Critical thinking: How do we honestly interpret a mixed result (e.g., 0/11 for direction, 7/11 for risk) without overclaiming?
```

---

## Phase 8 — Freeze Results, Figures & Manuscript Verification

```text
Follow Phase 8 of EXECUTION_GUIDE_8_PHASES.md.
1. Build and run src/make_figures.py to generate the 4 publication figures in figures/ (300 DPI, LKR units, clean labels) directly from results/summary_11_stocks.csv and results/per_stock/*.json.
2. Create paper/manuscript_checklist.md verifying every number in AIMT_abstract_fernando.pdf against our frozen JSON/CSV outputs.

At the end, explain in simple terms:
- Why generating charts and paper numbers directly from frozen JSON files prevents reviewer-flagged contradictions.
- What the core story of our finished study is in one clear paragraph I can present to my supervisor or conference audience.
- Critical thinking: What are the top 3 tough questions a reviewer or examiner will ask about this study, and how should I answer them honestly?
```

---

## Bonus: Final Supervisor / Viva Preparation Prompt

Use this after Phase 8 is complete:

```text
Act as a strict but fair conference reviewer and my research supervisor.
Review results/summary_11_stocks.csv and AIMT_abstract_fernando.pdf.
1. Give me a 2-minute beginner-friendly script to explain what we did from start to finish.
2. Ask me 5 realistic viva / Q&A questions about data leakage, naive baselines, HAR vs. Random Forest, Holm correction, and CSE market limitations — and provide strong, honest model answers for each.
3. List 3 research habits I should always apply in my next ML/finance project so I never repeat the mistakes from our first rejected abstract.
```
