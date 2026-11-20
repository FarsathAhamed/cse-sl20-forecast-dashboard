"""
Phase 7: Multi-Stock Results Aggregator & Holm-Bonferroni Adjuster.

Reads all 11 frozen JSON files from results/per_stock/, applies Holm-Bonferroni
multiple-testing adjustment across the 11 stocks for each hypothesis family,
and writes:
    - results/summary_11_stocks.csv
    - results/summary_11_stocks.json

Usage:
    python src/summarize_results.py --results-dir results/per_stock --out results/summary_11_stocks.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd

from stats_tests import holm_adjust


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 11-stock JSON results with Holm-Bonferroni adjustment.")
    parser.add_argument("--results-dir", default="results/per_stock", help="Directory of per-stock JSON files")
    parser.add_argument("--out", default="results/summary_11_stocks.csv", help="Output summary CSV path")
    parser.add_argument("--alpha", type=float, default=0.05, help="Family-wise significance level")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    json_files = sorted(res_dir.glob("*.json"))
    if len(json_files) != 11:
        raise RuntimeError(f"Expected 11 per-stock JSON files in {res_dir}, found {len(json_files)}")

    rows = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        d = data["direction"]
        r = data["risk"]
        rows.append(
            {
                "symbol": data["symbol"],
                "name": data["name"],
                "n_train_dir": d["n_train"],
                "n_test_dir": d["n_test"],
                "n_test_risk": r["n_test"],
                # Direction metrics
                "rf_train_acc": d["rf_train_acc"],
                "rf_acc": d["rf_test_acc"],
                "rf_acc_ci_low": d["rf_acc_ci_low"],
                "rf_acc_ci_high": d["rf_acc_ci_high"],
                "rf_bal_acc": d["rf_bal_acc"],
                "rf_mcc": d["rf_mcc"],
                "maj_acc": d["maj_rate_test"],
                "prev_acc": d["prev_day_acc"],
                "lr_acc": d["lr_acc"],
                "dir_p_raw": d["dir_p_raw"],
                # Risk metrics
                "har_r2_vs_hist": r["har_r2_vs_hist"],
                "har_dm_hist_stat": r["dm_har_vs_hist"]["dm_stat"],
                "har_dm_hist_p_raw": r["dm_har_vs_hist"]["p_raw"],
                "har_r2_vs_m22": r["har_r2_vs_m22"],
                "har_dm_m22_stat": r["dm_har_vs_m22"]["dm_stat"],
                "har_dm_m22_p_raw": r["dm_har_vs_m22"]["p_raw"],
                "rf_risk_r2_vs_hist": r["rf_r2_vs_hist"],
                "rf_dm_hist_p_raw": r["dm_rf_vs_hist"]["p_raw"],
                "har_vs_rf_dm_stat": r["dm_har_vs_rf"]["dm_stat"],
                "har_vs_rf_p_raw": r["dm_har_vs_rf"]["p_raw"],
                "spearman_har": r["spearman_har"],
                "spearman_rf": r["spearman_rf"],
            }
        )

    df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    # Apply Holm-Bonferroni adjustment across the 11 stocks for each hypothesis family
    df["dir_p_holm"] = holm_adjust(df["dir_p_raw"].tolist(), alpha=args.alpha)
    df["har_dm_hist_p_holm"] = holm_adjust(df["har_dm_hist_p_raw"].tolist(), alpha=args.alpha)
    df["har_dm_m22_p_holm"] = holm_adjust(df["har_dm_m22_p_raw"].tolist(), alpha=args.alpha)
    df["rf_dm_hist_p_holm"] = holm_adjust(df["rf_dm_hist_p_raw"].tolist(), alpha=args.alpha)
    df["har_vs_rf_p_holm"] = holm_adjust(df["har_vs_rf_p_raw"].tolist(), alpha=args.alpha)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_json = out_csv.with_suffix(".json")
    summary_meta = {
        "n_stocks": len(df),
        "alpha": args.alpha,
        "direction_sig_holm_count": int((df["dir_p_holm"] < args.alpha).sum()),
        "har_vs_hist_sig_holm_count": int((df["har_dm_hist_p_holm"] < args.alpha).sum()),
        "har_vs_m22_sig_holm_count": int((df["har_dm_m22_p_holm"] < args.alpha).sum()),
        "rf_risk_vs_hist_sig_holm_count": int((df["rf_dm_hist_p_holm"] < args.alpha).sum()),
        "har_beats_rf_sig_holm_count": int((df["har_vs_rf_p_holm"] < args.alpha).sum()),
        "rf_acc_range_pct": [round(float(df["rf_acc"].min() * 100), 2), round(float(df["rf_acc"].max() * 100), 2)],
        "har_sig_r2_range_pct": [
            round(float(df.loc[df["har_dm_hist_p_holm"] < args.alpha, "har_r2_vs_hist"].min() * 100), 2),
            round(float(df.loc[df["har_dm_hist_p_holm"] < args.alpha, "har_r2_vs_hist"].max() * 100), 2),
        ]
        if (df["har_dm_hist_p_holm"] < args.alpha).any()
        else None,
        "rows": df.to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    print(f"Saved master summary -> {out_csv}")
    print(f"Saved JSON metadata  -> {out_json}\n")

    print("=== H1: DIRECTION FORECASTING (11 S&P SL20 Stocks) ===")
    dir_cols = ["symbol", "n_test_dir", "rf_acc", "rf_bal_acc", "rf_mcc", "maj_acc", "prev_acc", "lr_acc", "dir_p_raw", "dir_p_holm"]
    print(df[dir_cols].to_string(index=False))

    print("\n=== H2: RISK / VOLATILITY FORECASTING (11 S&P SL20 Stocks) ===")
    risk_cols = [
        "symbol", "n_test_risk", "har_r2_vs_hist", "har_dm_hist_p_raw", "har_dm_hist_p_holm",
        "har_r2_vs_m22", "har_dm_m22_p_raw", "har_dm_m22_p_holm",
        "rf_risk_r2_vs_hist", "har_vs_rf_p_holm",
    ]
    print(df[risk_cols].to_string(index=False))

    print("\n=== HOLM-ADJUSTED SIGNIFICANCE COUNTS (alpha = 0.05) ===")
    print(f"  H1 Direction (RF > Majority):      {summary_meta['direction_sig_holm_count']} / 11")
    print(f"  H2 Risk (HAR > Historical Avg):    {summary_meta['har_vs_hist_sig_holm_count']} / 11  (sig R2 range: {summary_meta['har_sig_r2_range_pct']})")
    print(f"  H2 Risk (HAR > 22-Day Mean):       {summary_meta['har_vs_m22_sig_holm_count']} / 11")
    print(f"  H2 Risk (RF  > Historical Avg):    {summary_meta['rf_risk_vs_hist_sig_holm_count']} / 11")
    print(f"  H2 Risk (HAR > RF):                {summary_meta['har_beats_rf_sig_holm_count']} / 11")


if __name__ == "__main__":
    main()
