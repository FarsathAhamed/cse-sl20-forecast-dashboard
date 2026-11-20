"""
Phase 7: Master Pipeline Runner for all 11 S&P SL20 stocks.

Runs the exact same pre-registered protocol (configs/protocol.yaml) on each
stock's processed Direction and Risk datasets, computes statistical tests
(Wilson CI, exact binomial test, HAC Diebold-Mariano tests), and writes one
frozen JSON per stock to results/per_stock/<SYMBOL>.json.

Usage:
    python src/run_all_stocks.py --config configs/protocol.yaml
"""

import argparse
import json
from pathlib import Path
import time
import pandas as pd
import yaml

from models_direction import run_direction_models
from models_risk import run_risk_models
from stats_tests import binomial_test_gt, dm_test_hac, wilson_ci


def run_all(config_path: str, proc_dir: str, out_dir: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc_path = Path(proc_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    alpha_ci = cfg["statistics"].get("wilson_ci_alpha", 0.05)
    hac_lags = cfg["statistics"].get("hac_maxlags", None)

    t0 = time.time()
    for i, item in enumerate(cfg["tickers"], start=1):
        sym = item["symbol"]
        name = item["name"]
        print(f"[{i:2d}/11] Running {sym} ({name})...", flush=True)

        df_dir = pd.read_csv(proc_path / f"{sym}_dir.csv", parse_dates=["Date"])
        df_risk = pd.read_csv(proc_path / f"{sym}_risk.csv", parse_dates=["Date"])

        dir_res = run_direction_models(df_dir, cfg)
        risk_res = run_risk_models(df_risk, cfg)

        # Direction statistical tests
        ci_lo, ci_hi = wilson_ci(dir_res["rf_correct"], dir_res["n_test"], alpha=alpha_ci)
        dir_p_raw = binomial_test_gt(
            k=dir_res["rf_correct"],
            n=dir_res["n_test"],
            p0=dir_res["maj_rate_test"],
        )
        dir_res["rf_acc_ci_low"] = round(ci_lo, 4)
        dir_res["rf_acc_ci_high"] = round(ci_hi, 4)
        dir_res["dir_p_raw"] = round(dir_p_raw, 6)

        # Risk HAC Diebold-Mariano tests
        y_te_r = risk_res["_y_test"]
        risk_res["dm_har_vs_hist"] = dm_test_hac(
            y_te_r, risk_res["_y_pred_har"], risk_res["_y_pred_hist"], maxlags=hac_lags
        )
        risk_res["dm_har_vs_m22"] = dm_test_hac(
            y_te_r, risk_res["_y_pred_har"], risk_res["_y_pred_m22"], maxlags=hac_lags
        )
        risk_res["dm_rf_vs_hist"] = dm_test_hac(
            y_te_r, risk_res["_y_pred_rf"], risk_res["_y_pred_hist"], maxlags=hac_lags
        )
        risk_res["dm_har_vs_rf"] = dm_test_hac(
            y_te_r, risk_res["_y_pred_har"], risk_res["_y_pred_rf"], maxlags=hac_lags
        )

        # Strip private numpy arrays before JSON serialization
        dir_clean = {k: v for k, v in dir_res.items() if not k.startswith("_")}
        risk_clean = {k: v for k, v in risk_res.items() if not k.startswith("_")}

        payload = {
            "symbol": sym,
            "name": name,
            "direction": dir_clean,
            "risk": risk_clean,
        }
        out_file = out_path / f"{sym}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"        Dir RF={dir_clean['rf_test_acc']*100:.1f}% vs Maj={dir_clean['maj_rate_test']*100:.1f}% "
            f"(p={dir_clean['dir_p_raw']:.4f}) | "
            f"Risk HAR R2={risk_clean['har_r2_vs_hist']*100:.1f}% "
            f"(DM p={risk_clean['dm_har_vs_hist']['p_raw']:.4f})",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\nCompleted all 11 stocks in {elapsed:.1f}s -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full leakage-controlled evaluation on all 11 stocks.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--proc-dir", default="data/processed", help="Processed features folder")
    parser.add_argument("--out-dir", default="results/per_stock", help="Output folder for per-stock JSONs")
    args = parser.parse_args()
    run_all(args.config, args.proc_dir, args.out_dir)


if __name__ == "__main__":
    main()
