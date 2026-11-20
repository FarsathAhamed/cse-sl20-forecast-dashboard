"""
Phase 8: Publication-Ready Figure Generator (300 DPI, LKR units).

Generates 4 figures directly from frozen outputs (results/summary_11_stocks.csv,
results/per_stock/*.json, data/audit/cleaning_breaks.json, data/raw_cse/*.csv):
    1. figures/fig1_direction_vs_baselines.png
    2. figures/fig2_risk_har_oos_r2.png
    3. figures/fig3_price_series_breaks.png
    4. figures/fig4_permutation_importance.png

Usage:
    python src/make_figures.py --summary results/summary_11_stocks.csv --out-dir figures
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def fig1_direction_vs_baselines(df: pd.DataFrame, out_path: Path) -> None:
    df_s = df.sort_values("rf_acc", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_s))

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    rf_pct = df_s["rf_acc"] * 100.0
    ci_lo = (df_s["rf_acc"] - df_s["rf_acc_ci_low"]) * 100.0
    ci_hi = (df_s["rf_acc_ci_high"] - df_s["rf_acc"]) * 100.0

    ax.errorbar(
        rf_pct,
        y + 0.12,
        xerr=[ci_lo, ci_hi],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.4,
        capsize=3.5,
        markersize=6.5,
        label="Random Forest (95% Wilson CI)",
        zorder=4,
    )
    ax.scatter(
        df_s["maj_acc"] * 100.0,
        y - 0.12,
        marker="s",
        s=52,
        color="#d62728",
        label="Majority-Class Benchmark",
        zorder=5,
    )
    ax.scatter(
        df_s["prev_acc"] * 100.0,
        y - 0.12,
        marker="^",
        s=48,
        color="#2ca02c",
        alpha=0.85,
        label="Previous-Day Direction",
        zorder=3,
    )
    ax.scatter(
        df_s["lr_acc"] * 100.0,
        y + 0.12,
        marker="D",
        s=38,
        color="#ff7f0e",
        alpha=0.85,
        label="Logistic Regression",
        zorder=3,
    )

    ax.axvline(50.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, label="50% Coin-Flip Reference")
    ax.set_yticks(y)
    ax.set_yticklabels(df_s["symbol"])
    ax.set_xlabel("Hold-Out Classification Accuracy (%)")
    ax.set_title("H1: Next-Day Direction Accuracy vs. Naive Benchmarks (0 / 11 Significant, Holm p < 0.05)")
    ax.set_xlim(42, 69)
    ax.legend(loc="lower right", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_risk_har_oos_r2(df: pd.DataFrame, out_path: Path) -> None:
    df_s = df.sort_values("har_r2_vs_hist", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_s))
    bar_h = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.6), dpi=300, sharey=True)

    # Panel A: HAR vs RF (vs Historical Average)
    ax1.barh(
        y + bar_h / 2,
        df_s["har_r2_vs_hist"] * 100.0,
        height=bar_h,
        color="#1f77b4",
        label="HAR (OLS)",
    )
    ax1.barh(
        y - bar_h / 2,
        df_s["rf_risk_r2_vs_hist"] * 100.0,
        height=bar_h,
        color="#ff7f0e",
        alpha=0.85,
        label="Random Forest",
    )
    for idx, row in df_s.iterrows():
        val = row["har_r2_vs_hist"] * 100.0
        star = "*" if row["har_dm_hist_p_holm"] < 0.05 else ""
        ax1.text(val + 0.5, idx + bar_h / 2, f"{val:.1f}%{star}", va="center", fontsize=8.5, color="#1f77b4", fontweight="bold")

    ax1.axvline(0.0, color="#333333", linewidth=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df_s["symbol"])
    ax1.set_xlabel("Out-of-Sample R² vs. Historical Average (%)")
    ax1.set_title("A. Risk Forecast vs. Historical Average\n(HAR sig. in 11/11, *Holm p < 0.05)")
    ax1.set_xlim(0, 34)
    ax1.legend(loc="lower right", fontsize=8.8)

    # Panel B: HAR vs 22-Day Rolling Mean
    colors_m22 = ["#2ca02c" if p < 0.05 else "#999999" for p in df_s["har_dm_m22_p_holm"]]
    ax2.barh(y, df_s["har_r2_vs_m22"] * 100.0, height=0.55, color=colors_m22)
    for idx, row in df_s.iterrows():
        val = row["har_r2_vs_m22"] * 100.0
        star = "*" if row["har_dm_m22_p_holm"] < 0.05 else ""
        ax2.text(val + 0.4, idx, f"{val:.1f}%{star}", va="center", fontsize=8.5, fontweight="bold")

    ax2.axvline(0.0, color="#333333", linewidth=0.9)
    ax2.set_xlabel("Out-of-Sample R² vs. 22-Day Rolling Mean (%)")
    ax2.set_title("B. HAR Gain Over 22-Day Rolling Mean\n(Green = Holm p < 0.05: 7/11 stocks)")
    ax2.set_xlim(0, 25)

    fig.suptitle("H2: Next-Day Price Risk (High–Low Range / Close) Out-of-Sample Forecasting", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_price_series_breaks(
    raw_dir: Path,
    breaks_json: Path,
    per_stock_dir: Path,
    out_path: Path,
) -> None:
    with open(breaks_json, "r", encoding="utf-8") as f:
        breaks_all = json.load(f)

    # 4 representative stocks showing splits, closures, and liquidity regimes
    reps = [
        ("JKH", "JKH_N0000.csv", "JKH (1:10 Split Oct 2024)"),
        ("HHL", "HHL_N0000.csv", "HHL (1:5 Split Apr 2025)"),
        ("DIAL", "DIAL_N0000.csv", "DIAL (High Flat-Day Share & Rally)"),
        ("HAYL", "HAYL_N0000.csv", "HAYL (2021 Split Halt & Regime Shift)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), dpi=300)
    for ax, (sym, fname, title) in zip(axes.flat, reps):
        df = pd.read_csv(raw_dir / fname, parse_dates=["Date"]).sort_values("Date")
        # Restrict visual to active trading window up to 2026-02-17 (exclude frozen Yahoo tail)
        df = df[df["Date"] <= "2026-02-17"]

        with open(per_stock_dir / f"{sym}.json", "r", encoding="utf-8") as f:
            stock_meta = json.load(f)["direction"]
        test_start = pd.Timestamp(stock_meta["test_start"])
        test_end = pd.Timestamp(stock_meta["test_end"])

        ax.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1.1, label="Close (LKR)")
        ax.axvspan(test_start, test_end, color="#ffe8cc", alpha=0.55, label="Hold-Out Test Window")

        # Plot jump breaks (red solid) and gap breaks (gray dashed)
        jump_labeled = False
        gap_labeled = False
        for ev in breaks_all.get(sym, []):
            dt = pd.Timestamp(ev["date"])
            if "jump" in ev["reason"]:
                ax.axvline(
                    dt,
                    color="#d62728",
                    linestyle="-",
                    linewidth=1.3,
                    alpha=0.9,
                    label="Split / Jump Break (≥50%)" if not jump_labeled else None,
                )
                jump_labeled = True
            else:
                ax.axvline(
                    dt,
                    color="#7f7f7f",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                    label="Closure / Halt Break (≥6d)" if not gap_labeled else None,
                )
                gap_labeled = True

        ax.set_title(title)
        ax.set_ylabel("Close Price (LKR)")
        ax.legend(loc="upper left", fontsize=7.8)

    fig.suptitle("Representative CSE Price Series (LKR), Detected Structural Breaks & Chronological Hold-Out", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_permutation_importance(per_stock_dir: Path, out_path: Path) -> None:
    json_files = sorted(per_stock_dir.glob("*.json"))
    records = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        sym = data["symbol"]
        for feat, vals in data["direction"]["permutation_importance"].items():
            records.append({"symbol": sym, "feature": feat, "mean": vals["mean"]})

    imp_df = pd.DataFrame(records)
    agg = (
        imp_df.groupby("feature")["mean"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    y = np.arange(len(agg))
    colors = ["#1f77b4" if m > 0 else "#999999" for m in agg["mean"]]

    ax.barh(y, agg["mean"], xerr=agg["std"], height=0.6, color=colors, ecolor="#333333", capsize=3, alpha=0.85)
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.1, label="Zero Importance (Noise Baseline)")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["feature"])
    ax.set_xlabel("Hold-Out Permutation Importance (Δ Balanced Accuracy, Mean ± SD across 11 Stocks)")
    ax.set_title("Hold-Out Direction Permutation Importance: Technical Indicators Cluster Near Zero")
    ax.legend(loc="lower right", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: Generate publication figures from frozen results.")
    parser.add_argument("--summary", default="results/summary_11_stocks.csv", help="Master summary CSV")
    parser.add_argument("--per-stock-dir", default="results/per_stock", help="Per-stock JSON directory")
    parser.add_argument("--raw-dir", default="data/raw_cse", help="Raw CSV directory")
    parser.add_argument("--breaks-json", default="data/audit/cleaning_breaks.json", help="Cleaning breaks JSON")
    parser.add_argument("--out-dir", default="figures", help="Output figures directory")
    args = parser.parse_args()

    set_pub_style()
    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = out_dir / "fig1_direction_vs_baselines.png"
    f2 = out_dir / "fig2_risk_har_oos_r2.png"
    f3 = out_dir / "fig3_price_series_breaks.png"
    f4 = out_dir / "fig4_permutation_importance.png"

    fig1_direction_vs_baselines(df, f1)
    fig2_risk_har_oos_r2(df, f2)
    fig3_price_series_breaks(Path(args.raw_dir), Path(args.breaks_json), Path(args.per_stock_dir), f3)
    fig4_permutation_importance(Path(args.per_stock_dir), f4)

    print("Generated all 4 publication figures (300 DPI, LKR units):")
    for fp in [f1, f2, f3, f4]:
        print(f"  - {fp} ({fp.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
