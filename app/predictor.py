"""
Leakage-controlled inference engine for the CSE S&P SL20 Forecast Dashboard.

Reads frozen research artifacts (configs/protocol.yaml, data/raw_cse/,
data/processed/, results/per_stock/*.json, results/summary_11_stocks.csv)
in strictly read-only mode and caches fitted models under artifacts/models/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.cleaning import clean_stock_df  # noqa: E402
from src.split import chronological_split  # noqa: E402

PROTOCOL_PATH = ROOT_DIR / "configs" / "protocol.yaml"
RAW_DIR = ROOT_DIR / "data" / "raw_cse"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PER_STOCK_RESULTS_DIR = ROOT_DIR / "results" / "per_stock"
SUMMARY_CSV_PATH = ROOT_DIR / "results" / "summary_11_stocks.csv"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"


def load_protocol() -> dict[str, Any]:
    """Load the frozen pre-registered research protocol."""
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stock_catalog() -> list[dict[str, str]]:
    """Return the 11 S&P SL20 tickers, company names, and raw file stems."""
    cfg = load_protocol()
    return [dict(item) for item in cfg["tickers"]]


def _get_ticker_meta(symbol: str) -> dict[str, str]:
    symbol_upper = symbol.strip().upper()
    for item in get_stock_catalog():
        if item["symbol"].upper() == symbol_upper:
            return item
    raise ValueError(f"Unknown ticker '{symbol}'. Expected one of the 11 S&P SL20 stocks.")


def load_frozen_stock_results(symbol: str) -> dict[str, Any]:
    """Load frozen hold-out evaluation JSON and summary row for a stock."""
    meta = _get_ticker_meta(symbol)
    sym = meta["symbol"]
    json_path = PER_STOCK_RESULTS_DIR / f"{sym}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        per_stock = json.load(f)

    summary_row: dict[str, Any] = {}
    if SUMMARY_CSV_PATH.exists():
        summary_df = pd.read_csv(SUMMARY_CSV_PATH)
        match = summary_df.loc[summary_df["symbol"] == sym]
        if not match.empty:
            summary_row = match.iloc[0].to_dict()

    return {
        "per_stock": per_stock,
        "summary_row": summary_row,
    }


def load_all_stocks_summary() -> pd.DataFrame:
    """Load the frozen 11-stock benchmark summary table."""
    return pd.read_csv(SUMMARY_CSV_PATH)


def load_price_history(symbol: str) -> pd.DataFrame:
    """Load and clean OHLCV price history from data/raw_cse/<file_stem>.csv for charting."""
    cfg = load_protocol()
    meta = _get_ticker_meta(symbol)
    raw_path = RAW_DIR / f"{meta['file_stem']}.csv"
    df_raw = pd.read_csv(raw_path)
    df_clean, _ = clean_stock_df(df_raw, cfg)
    return df_clean


def train_or_load_models(symbol: str, force_retrain: bool = False) -> dict[str, Any]:
    """
    Fit (on training split only) or load cached Direction RF & HAR Risk models.

    Uses frozen best_params from results/per_stock/<symbol>.json and the exact
    chronological split from src/split.py so no hold-out or future data leaks.
    """
    cfg = load_protocol()
    meta = _get_ticker_meta(symbol)
    sym = meta["symbol"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS_DIR / f"{sym}_models.joblib"

    if bundle_path.exists() and not force_retrain:
        return joblib.load(bundle_path)

    frozen = load_frozen_stock_results(sym)["per_stock"]
    dir_feat_cols = cfg["features"]["direction"]
    har_feat_cols = cfg["features"]["risk_har"]
    split_cfg = cfg["split"]
    dir_cfg = cfg["direction_model"]

    df_dir = pd.read_csv(PROCESSED_DIR / f"{sym}_dir.csv", parse_dates=["Date"])
    df_risk = pd.read_csv(PROCESSED_DIR / f"{sym}_risk.csv", parse_dates=["Date"])

    train_dir, _, _ = chronological_split(
        df_dir,
        train_frac=split_cfg["train_fraction"],
        gap=split_cfg["embargo_gap_days"],
    )
    train_risk, _, _ = chronological_split(
        df_risk,
        train_frac=split_cfg["train_fraction"],
        gap=split_cfg["embargo_gap_days"],
    )

    best_params = frozen["direction"]["best_params"]
    rf_dir = RandomForestClassifier(
        random_state=dir_cfg["random_state"],
        n_jobs=-1,
        **best_params,
    )
    rf_dir.fit(train_dir[dir_feat_cols], train_dir["y_dir"].astype(int).to_numpy())

    har_risk = LinearRegression()
    har_risk.fit(train_risk[har_feat_cols], train_risk["y_risk"].to_numpy(dtype=float))

    # Training-only risk quantiles for Low / Medium / High tiering
    y_risk_train = train_risk["y_risk"].to_numpy(dtype=float)
    q33, q67 = np.quantile(y_risk_train, [1.0 / 3.0, 2.0 / 3.0])

    bundle = {
        "symbol": sym,
        "name": meta["name"],
        "direction_model": rf_dir,
        "risk_model": har_risk,
        "direction_features": dir_feat_cols,
        "risk_features": har_feat_cols,
        "risk_quantiles": {"q33": float(q33), "q67": float(q67)},
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def _classify_risk_level(pred_range: float, quantiles: dict[str, float]) -> str:
    if pred_range < quantiles["q33"]:
        return "Low"
    if pred_range < quantiles["q67"]:
        return "Medium"
    return "High"


def get_latest_forecast(symbol: str) -> dict[str, Any]:
    """
    Compute the latest next-day direction & risk forecast for a stock.

    Returns price snapshot, direction label ('UP' / 'NOT-UP'), P(UP),
    confidence score, HAR expected next-day range (%), risk tier, and
    frozen hold-out benchmark metrics.
    """
    meta = _get_ticker_meta(symbol)
    sym = meta["symbol"]
    bundle = train_or_load_models(sym)
    frozen = load_frozen_stock_results(sym)

    df_price = load_price_history(sym)
    latest_price_row = df_price.iloc[-1]
    prev_price_row = df_price.iloc[-2] if len(df_price) >= 2 else latest_price_row

    latest_close = float(latest_price_row["Close"])
    prev_close = float(prev_price_row["Close"])
    daily_change_pct = ((latest_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

    df_dir = pd.read_csv(PROCESSED_DIR / f"{sym}_dir.csv", parse_dates=["Date"])
    df_risk = pd.read_csv(PROCESSED_DIR / f"{sym}_risk.csv", parse_dates=["Date"])

    latest_dir_row = df_dir.iloc[[-1]]
    latest_risk_row = df_risk.iloc[[-1]]

    X_dir_latest = latest_dir_row[bundle["direction_features"]]
    X_risk_latest = latest_risk_row[bundle["risk_features"]]

    probs = bundle["direction_model"].predict_proba(X_dir_latest)[0]
    classes = list(bundle["direction_model"].classes_)
    prob_up = float(probs[classes.index(1)]) if 1 in classes else float(probs[-1])
    direction_label = "UP" if prob_up >= 0.5 else "NOT-UP"
    confidence_score = max(prob_up, 1.0 - prob_up)

    pred_risk_range = float(bundle["risk_model"].predict(X_risk_latest)[0])
    pred_risk_range = max(pred_risk_range, 0.0)
    risk_level = _classify_risk_level(pred_risk_range, bundle["risk_quantiles"])

    latest_feature_date = pd.to_datetime(latest_dir_row["Date"].iloc[0]).strftime("%Y-%m-%d")
    latest_price_date = pd.to_datetime(latest_price_row["Date"]).strftime("%Y-%m-%d")

    return {
        "symbol": sym,
        "name": meta["name"],
        "latest_date": latest_price_date,
        "latest_feature_date": latest_feature_date,
        "latest_close_lkr": round(latest_close, 2),
        "daily_change_pct": round(daily_change_pct, 2),
        "direction_label": direction_label,
        "prob_up": round(prob_up, 4),
        "confidence_score": round(confidence_score, 4),
        "predicted_risk_range": round(pred_risk_range, 6),
        "predicted_risk_range_pct": round(pred_risk_range * 100.0, 2),
        "risk_level": risk_level,
        "frozen_metrics": frozen["per_stock"],
        "summary_row": frozen["summary_row"],
    }


if __name__ == "__main__":
    for test_sym in ["JKH", "COMB"]:
        fc = get_latest_forecast(test_sym)
        print(
            f"{fc['symbol']} ({fc['name']}) | Date: {fc['latest_date']} | "
            f"Close: LKR {fc['latest_close_lkr']:.2f} ({fc['daily_change_pct']:+.2f}%) | "
            f"Next-Day Dir: {fc['direction_label']} (P(UP)={fc['prob_up']:.1%}, "
            f"Conf={fc['confidence_score']:.1%}) | "
            f"HAR Risk Range: {fc['predicted_risk_range_pct']:.2f}% [{fc['risk_level']}]"
        )
