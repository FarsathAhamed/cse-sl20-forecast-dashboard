"""
Phase 4: Stationary Feature & Target Engineering (Leakage-Controlled).

Computes strictly causal features (using data up to day t only) and next-day
targets (day t+1) strictly INSIDE each continuous regime_id so no rolling
indicator or target ever crosses a structural break.

Outputs:
    data/processed/<SYMBOL>_dir.csv
    data/processed/<SYMBOL>_risk.csv
    data/processed/feature_summary.csv

Usage:
    python src/features.py --config configs/protocol.yaml
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from cleaning import clean_stock_df

RAW_FORBIDDEN_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}


def _compute_regime_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    c = g["Close"]
    h = g["High"]
    l = g["Low"]
    v = g["Volume"]

    # --- 12 Stationary Direction Features (strictly <= day t) ---
    g["ret1"] = np.log(c).diff()
    g["ret_lag2"] = g["ret1"].shift(1)
    g["ret_lag3"] = g["ret1"].shift(2)
    g["ret_lag5"] = g["ret1"].shift(4)
    g["roc10"] = c.pct_change(10)
    g["ma10_rel"] = c / c.rolling(10).mean() - 1.0
    g["ma30_rel"] = c / c.rolling(30).mean() - 1.0

    diff = c.diff()
    gain = diff.clip(lower=0.0).rolling(14).mean()
    loss = (-diff.clip(upper=0.0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # If loss == 0 and gain > 0 -> RSI = 100; if both == 0 -> RSI = 50
    rsi = np.where((loss == 0.0) & (gain > 0.0), 100.0, np.where((loss == 0.0) & (gain == 0.0), 50.0, rsi))
    g["rsi"] = rsi

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    g["macd_rel"] = (ema12 - ema26) / c

    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    denom = (hi14 - lo14).replace(0.0, np.nan)
    g["stoch_k"] = 100.0 * (c - lo14) / denom
    g["willr"] = -100.0 * (hi14 - c) / denom

    vol_ma20 = v.rolling(20).mean().replace(0.0, np.nan)
    g["vol_rel"] = np.log(v.replace(0.0, np.nan) / vol_ma20)

    # --- 3 HAR Risk Features (strictly <= day t) ---
    hl_range = (h - l) / c
    g["hl_range"] = hl_range
    g["har_d"] = hl_range
    g["har_w"] = hl_range.rolling(5).mean()
    g["har_m"] = hl_range.rolling(22).mean()

    # --- Next-Day Targets (strictly within this regime_id) ---
    next_c = c.shift(-1)
    g["y_dir"] = (next_c > c).astype(float).where(next_c.notna())
    g["y_risk"] = hl_range.shift(-1)
    g["y_absret"] = g["ret1"].shift(-1).abs()

    # Mark last row of this regime so we can verify no target crosses a break
    g["is_regime_last_row"] = False
    if len(g) > 0:
        g.iloc[-1, g.columns.get_loc("is_regime_last_row")] = True

    return g


def build_features_for_stock(raw_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dir_feats = cfg["features"]["direction"]
    har_feats = cfg["features"]["risk_har"]

    # Gate 4 check: no raw price/volume levels allowed in features
    assert not (set(dir_feats) & RAW_FORBIDDEN_COLS), "Raw price/volume column found in direction features!"
    assert not (set(har_feats) & RAW_FORBIDDEN_COLS), "Raw price/volume column found in HAR features!"

    clean_df, clean_meta = clean_stock_df(raw_df, cfg)

    # Compute features & targets strictly inside each continuous regime_id
    parts = [_compute_regime_features(grp) for _, grp in clean_df.groupby("regime_id", sort=True)]
    feat_df = pd.concat(parts, axis=0).sort_values("Date").reset_index(drop=True)

    # Exclude post-break purge window rows
    valid_df = feat_df.loc[~feat_df["in_purge_window"]].copy()
    valid_df = valid_df.replace([np.inf, -np.inf], np.nan)

    # Direction dataset
    dir_cols = ["Date", "regime_id", "is_regime_last_row"] + dir_feats + ["y_dir"]
    df_dir = valid_df[dir_cols].dropna(subset=dir_feats + ["y_dir"]).reset_index(drop=True)
    df_dir["y_dir"] = df_dir["y_dir"].astype(int)

    # Risk dataset (keep both HAR features and direction features for RF risk comparison if needed)
    risk_cols = ["Date", "regime_id", "is_regime_last_row"] + har_feats + ["y_risk", "y_absret"]
    df_risk = valid_df[risk_cols].dropna(subset=har_feats + ["y_risk"]).reset_index(drop=True)

    # --- Gate 4 Leakage Unit Tests ---
    for name, d, fcols, ycol in [
        ("dir", df_dir, dir_feats, "y_dir"),
        ("risk", df_risk, har_feats, "y_risk"),
    ]:
        assert d["Date"].is_monotonic_increasing, f"{name}: Dates not monotonic!"
        assert not d["is_regime_last_row"].any(), f"{name}: Target crossed a structural break!"
        assert not d[fcols + [ycol]].isna().any().any(), f"{name}: NaNs remain!"
        assert np.isfinite(d[fcols + [ycol]].to_numpy()).all(), f"{name}: Infs remain!"

    # Verify y_dir is NOT same-day direction (ret1 > 0)
    same_day_agreement = float(((df_dir["ret1"] > 0).astype(int) == df_dir["y_dir"]).mean())
    assert same_day_agreement < 0.85, f"Suspiciously high same-day agreement ({same_day_agreement:.3f}) — check shift(-1)!"

    # Verify y_risk is strictly non-negative
    assert (df_risk["y_risk"] >= 0.0).all(), "Negative risk target detected!"

    # Drop helper column before saving
    df_dir = df_dir.drop(columns=["is_regime_last_row"])
    df_risk = df_risk.drop(columns=["is_regime_last_row"])

    summary = {
        "n_dir_rows": len(df_dir),
        "dir_start": df_dir["Date"].min().strftime("%Y-%m-%d"),
        "dir_end": df_dir["Date"].max().strftime("%Y-%m-%d"),
        "dir_up_rate_pct": round(float(df_dir["y_dir"].mean() * 100.0), 2),
        "same_day_ret1_agree_pct": round(same_day_agreement * 100.0, 2),
        "n_risk_rows": len(df_risk),
        "risk_mean_hl_pct": round(float(df_risk["y_risk"].mean() * 100.0), 3),
        "n_breaks": clean_meta["n_breaks"],
    }
    return df_dir, df_risk, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Build stationary features & targets for all 11 stocks.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--raw-dir", default="data/raw_cse", help="Raw CSV folder")
    parser.add_argument("--out-dir", default="data/processed", help="Processed feature output folder")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in cfg["tickers"]:
        sym = item["symbol"]
        raw_df = pd.read_csv(raw_dir / f"{item['file_stem']}.csv")
        df_dir, df_risk, summary = build_features_for_stock(raw_df, cfg)

        df_dir.to_csv(out_dir / f"{sym}_dir.csv", index=False)
        df_risk.to_csv(out_dir / f"{sym}_risk.csv", index=False)
        rows.append({"symbol": sym} | summary)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "feature_summary.csv", index=False)

    print("Gate 4 Leakage Unit Tests PASSED for all 11 stocks!\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
