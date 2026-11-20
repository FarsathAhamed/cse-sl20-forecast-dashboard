"""
Phase 5: Leakage-Controlled Chronological Split with Embargo Gap.

Splits processed Direction and Risk datasets strictly chronologically:
    - Training set: first 80% minus 10-trading-day embargo gap
    - Embargo gap: 10 trading days strictly excluded between train and test
    - Hold-out test set: final 20% (untouched until final evaluation)

Usage:
    python src/split.py --config configs/protocol.yaml
"""

import argparse
from pathlib import Path
import pandas as pd
import yaml


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    gap: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    assert df["Date"].is_monotonic_increasing, "DataFrame must be strictly sorted by Date!"

    n = len(df)
    cut = int(n * train_frac)
    if cut - gap <= 0 or cut >= n:
        raise ValueError(f"Invalid split sizes for n={n}, train_frac={train_frac}, gap={gap}")

    train = df.iloc[: cut - gap].copy().reset_index(drop=True)
    embargo = df.iloc[cut - gap : cut].copy().reset_index(drop=True)
    test = df.iloc[cut:].copy().reset_index(drop=True)

    # Gate 5 strict anti-leakage assertions
    assert len(embargo) == gap, f"Expected {gap} embargo rows, got {len(embargo)}"
    assert train["Date"].max() < embargo["Date"].min(), "Train overlaps embargo!"
    assert embargo["Date"].max() < test["Date"].min(), "Embargo overlaps test!"
    assert train["Date"].max() < test["Date"].min(), "Chronological leakage: train_max >= test_min!"
    assert set(train["Date"]).isdisjoint(set(test["Date"])), "Duplicate dates across train/test!"

    return train, test, embargo


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Verify chronological 80/20 + 10-day gap split.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--proc-dir", default="data/processed", help="Processed features folder")
    parser.add_argument("--out-csv", default="data/audit/split_summary.csv", help="Output summary CSV")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_frac = cfg["split"]["train_fraction"]
    gap = cfg["split"]["embargo_gap_days"]
    proc_dir = Path(args.proc_dir)

    rows = []
    for item in cfg["tickers"]:
        sym = item["symbol"]
        df_dir = pd.read_csv(proc_dir / f"{sym}_dir.csv", parse_dates=["Date"])
        df_risk = pd.read_csv(proc_dir / f"{sym}_risk.csv", parse_dates=["Date"])

        tr_d, te_d, emb_d = chronological_split(df_dir, train_frac=train_frac, gap=gap)
        tr_r, te_r, emb_r = chronological_split(df_risk, train_frac=train_frac, gap=gap)

        rows.append(
            {
                "symbol": sym,
                "n_train_dir": len(tr_d),
                "train_start": tr_d["Date"].min().strftime("%Y-%m-%d"),
                "train_end": tr_d["Date"].max().strftime("%Y-%m-%d"),
                "gap_rows": len(emb_d),
                "test_start": te_d["Date"].min().strftime("%Y-%m-%d"),
                "test_end": te_d["Date"].max().strftime("%Y-%m-%d"),
                "n_test_dir": len(te_d),
                "n_train_risk": len(tr_r),
                "n_test_risk": len(te_r),
                "train_up_pct": round(float(tr_d["y_dir"].mean() * 100.0), 2),
                "test_up_pct": round(float(te_d["y_dir"].mean() * 100.0), 2),
            }
        )

    summary_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)

    print("Gate 5 Strict Chronological Split Assertions PASSED for all 11 stocks!\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
