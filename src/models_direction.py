"""
Phase 6A: Direction Forecasting Models & Naive Benchmarks (H1).

Strictly leakage-controlled:
- RandomForestClassifier tuned via GridSearchCV + TimeSeriesSplit(5) on TRAIN ONLY.
- LogisticRegression standardized via Pipeline(StandardScaler, LogisticRegression) on TRAIN ONLY.
- Naive benchmarks: Majority-Class Rule & Previous-Day Direction Rule.
- Hold-out test set evaluated ONCE; permutation importance computed on hold-out.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from split import chronological_split


def run_direction_models(df_dir: pd.DataFrame, cfg: dict) -> dict:
    feat_cols = cfg["features"]["direction"]
    split_cfg = cfg["split"]
    dir_cfg = cfg["direction_model"]

    train, test, embargo = chronological_split(
        df_dir,
        train_frac=split_cfg["train_fraction"],
        gap=split_cfg["embargo_gap_days"],
    )

    X_tr, y_tr = train[feat_cols], train["y_dir"].astype(int).to_numpy()
    X_te, y_te = test[feat_cols], test["y_dir"].astype(int).to_numpy()

    # 1. Random Forest tuned strictly inside training data via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=split_cfg["cv_splits"])
    param_grid = dir_cfg["rf_param_grid"]
    n_grid_combos = int(np.prod([len(v) for v in param_grid.values()]))

    grid = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=dir_cfg["random_state"],
            n_jobs=-1,
        ),
        param_grid=param_grid,
        cv=tscv,
        scoring=dir_cfg["scoring"],
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_tr, y_tr)
    rf = grid.best_estimator_

    # Evaluate once on hold-out
    y_pred_rf = rf.predict(X_te)
    y_pred_rf_train = rf.predict(X_tr)

    # 2. Naive Benchmarks
    # (a) Majority class learned from training set
    maj_class_train = int(y_tr.mean() >= 0.5)
    y_pred_maj_train = np.full(len(y_te), maj_class_train, dtype=int)
    maj_acc_train_rule = float(accuracy_score(y_te, y_pred_maj_train))
    # Also record actual hold-out majority prevalence (stricter null benchmark p0)
    maj_rate_test = float(max(y_te.mean(), 1.0 - y_te.mean()))

    # (b) Previous-day direction: predict tomorrow Up iff today's return (ret1) > 0
    y_pred_prev = (test["ret1"].to_numpy() > 0.0).astype(int)

    # (c) Standardized Logistic Regression (fit strictly on training data)
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=dir_cfg["random_state"])),
        ]
    )
    lr_pipe.fit(X_tr, y_tr)
    y_pred_lr = lr_pipe.predict(X_te)

    # 3. Hold-out Permutation Importance (30 repeats)
    perm = permutation_importance(
        rf,
        X_te,
        y_te,
        n_repeats=30,
        random_state=dir_cfg["random_state"],
        scoring=dir_cfg["scoring"],
        n_jobs=-1,
    )
    perm_imp = {
        f: {
            "mean": round(float(m), 5),
            "std": round(float(s), 5),
        }
        for f, m, s in zip(feat_cols, perm.importances_mean, perm.importances_std)
    }

    # Confusion matrix: [[TN, FP], [FN, TP]] for labels [0 (Not-Up), 1 (Up)]
    cm = confusion_matrix(y_te, y_pred_rf, labels=[0, 1])

    # Total trial count for this stock's direction track
    # n_grid_combos CV configs + 1 LogReg + 2 naive rules
    trials = {
        "rf_grid_configs": n_grid_combos,
        "rf_cv_fits": n_grid_combos * split_cfg["cv_splits"] + 1,
        "other_models": 3,  # Majority, Prev-Day, LogReg
    }

    return {
        "n_train": int(len(train)),
        "n_embargo": int(len(embargo)),
        "n_test": int(len(test)),
        "train_start": train["Date"].min().strftime("%Y-%m-%d"),
        "train_end": train["Date"].max().strftime("%Y-%m-%d"),
        "test_start": test["Date"].min().strftime("%Y-%m-%d"),
        "test_end": test["Date"].max().strftime("%Y-%m-%d"),
        "best_params": grid.best_params_,
        "cv_best_balanced_acc": round(float(grid.best_score_), 4),
        "rf_train_acc": round(float(accuracy_score(y_tr, y_pred_rf_train)), 4),
        "rf_test_acc": round(float(accuracy_score(y_te, y_pred_rf)), 4),
        "rf_correct": int((y_pred_rf == y_te).sum()),
        "rf_bal_acc": round(float(balanced_accuracy_score(y_te, y_pred_rf)), 4),
        "rf_mcc": round(float(matthews_corrcoef(y_te, y_pred_rf)), 4),
        "maj_class_from_train": maj_class_train,
        "maj_acc_train_rule": round(maj_acc_train_rule, 4),
        "maj_rate_test": round(maj_rate_test, 4),
        "prev_day_acc": round(float(accuracy_score(y_te, y_pred_prev)), 4),
        "prev_day_bal_acc": round(float(balanced_accuracy_score(y_te, y_pred_prev)), 4),
        "prev_day_mcc": round(float(matthews_corrcoef(y_te, y_pred_prev)), 4),
        "lr_acc": round(float(accuracy_score(y_te, y_pred_lr)), 4),
        "lr_bal_acc": round(float(balanced_accuracy_score(y_te, y_pred_lr)), 4),
        "lr_mcc": round(float(matthews_corrcoef(y_te, y_pred_lr)), 4),
        "confusion_matrix_labels": ["Not-Up (0)", "Up (1)"],
        "confusion_matrix": cm.tolist(),
        "permutation_importance": perm_imp,
        "trials": trials,
        # Arrays kept for Phase 7 statistical tests (not dumped raw to summary)
        "_y_test": y_te,
        "_y_pred_rf": y_pred_rf,
    }
