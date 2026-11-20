"""
Phase 6B: Risk / Volatility Forecasting Models & Naive Benchmarks (H2).

Target: next-day high-low price range relative to close:
    R_{t+1} = (High_{t+1} - Low_{t+1}) / Close_{t+1}

Models (strictly fit/tuned on TRAIN ONLY, evaluated ONCE on hold-out):
- HAR (Heterogeneous Autoregressive OLS on har_d, har_w, har_m)
- RandomForestRegressor (GridSearchCV + TimeSeriesSplit(5) on har_d, har_w, har_m)

Benchmarks:
- Historical Average (training-period mean of y_risk)
- 22-Day Rolling Mean (har_m at day t)
- Previous-Day Range (har_d at day t, reference)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from split import chronological_split


def oos_r2(y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_bench: np.ndarray) -> float:
    sse_model = float(np.sum((y_true - y_pred_model) ** 2))
    sse_bench = float(np.sum((y_true - y_pred_bench) ** 2))
    if sse_bench == 0.0:
        return float("nan")
    return 1.0 - sse_model / sse_bench


def run_risk_models(df_risk: pd.DataFrame, cfg: dict) -> dict:
    har_cols = cfg["features"]["risk_har"]
    split_cfg = cfg["split"]
    risk_cfg = cfg["risk_model"]

    train, test, embargo = chronological_split(
        df_risk,
        train_frac=split_cfg["train_fraction"],
        gap=split_cfg["embargo_gap_days"],
    )

    X_tr, y_tr = train[har_cols], train["y_risk"].to_numpy(dtype=float)
    X_te, y_te = test[har_cols], test["y_risk"].to_numpy(dtype=float)

    # 1. HAR (OLS) fit strictly on training data
    har = LinearRegression()
    har.fit(X_tr, y_tr)
    y_pred_har = har.predict(X_te)

    # 2. Random Forest Regressor tuned strictly inside training data via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=split_cfg["cv_splits"])
    param_grid = risk_cfg["rf_param_grid"]
    n_grid_combos = int(np.prod([len(v) for v in param_grid.values()]))

    grid = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=risk_cfg["random_state"],
            n_jobs=-1,
        ),
        param_grid=param_grid,
        cv=tscv,
        scoring=risk_cfg["scoring"],
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_tr, y_tr)
    rf = grid.best_estimator_
    y_pred_rf = rf.predict(X_te)

    # 3. Naive Benchmarks
    hist_mean = float(y_tr.mean())
    y_pred_hist = np.full(len(y_te), hist_mean, dtype=float)
    y_pred_m22 = test["har_m"].to_numpy(dtype=float)
    y_pred_prev = test["har_d"].to_numpy(dtype=float)

    # Metrics
    r2_har_hist = oos_r2(y_te, y_pred_har, y_pred_hist)
    r2_har_m22 = oos_r2(y_te, y_pred_har, y_pred_m22)
    r2_rf_hist = oos_r2(y_te, y_pred_rf, y_pred_hist)
    r2_rf_m22 = oos_r2(y_te, y_pred_rf, y_pred_m22)
    r2_m22_hist = oos_r2(y_te, y_pred_m22, y_pred_hist)
    r2_prev_hist = oos_r2(y_te, y_pred_prev, y_pred_hist)

    spearman_har = float(spearmanr(y_te, y_pred_har).statistic)
    spearman_rf = float(spearmanr(y_te, y_pred_rf).statistic)
    spearman_m22 = float(spearmanr(y_te, y_pred_m22).statistic)

    trials = {
        "rf_grid_configs": n_grid_combos,
        "rf_cv_fits": n_grid_combos * split_cfg["cv_splits"] + 1,
        "other_models": 4,  # HAR OLS, Historical Avg, 22d Mean, Prev-Day
    }

    return {
        "n_train": int(len(train)),
        "n_embargo": int(len(embargo)),
        "n_test": int(len(test)),
        "train_mean_range": round(hist_mean, 6),
        "test_mean_range": round(float(y_te.mean()), 6),
        "har_intercept": round(float(har.intercept_), 6),
        "har_coefs": {c: round(float(v), 6) for c, v in zip(har_cols, har.coef_)},
        "rf_best_params": grid.best_params_,
        "har_r2_vs_hist": round(float(r2_har_hist), 4),
        "har_r2_vs_m22": round(float(r2_har_m22), 4),
        "rf_r2_vs_hist": round(float(r2_rf_hist), 4),
        "rf_r2_vs_m22": round(float(r2_rf_m22), 4),
        "m22_r2_vs_hist": round(float(r2_m22_hist), 4),
        "prev_r2_vs_hist": round(float(r2_prev_hist), 4),
        "spearman_har": round(spearman_har, 4),
        "spearman_rf": round(spearman_rf, 4),
        "spearman_m22": round(spearman_m22, 4),
        "trials": trials,
        # Arrays kept for Phase 7 HAC Diebold-Mariano tests
        "_y_test": y_te,
        "_y_pred_har": y_pred_har,
        "_y_pred_rf": y_pred_rf,
        "_y_pred_hist": y_pred_hist,
        "_y_pred_m22": y_pred_m22,
        "_y_pred_prev": y_pred_prev,
    }
