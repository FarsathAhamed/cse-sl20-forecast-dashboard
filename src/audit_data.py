"""
Phase 2: Pre-Modeling Data Quality Audit for all 11 S&P SL20 stocks.

Inspects raw CSVs in data/raw_cse/ before any cleaning or modeling:
- Row counts and date ranges
- Zero/non-positive OHLC placeholder rows
- Zero-volume rows
- Exact stale/copied consecutive OHLCV rows
- Flat-day percentage (Close[t+1] == Close[t])
- Calendar gaps >= break_min_calendar_gap_days (default 6 days)
- Single-day price jumps >= break_min_abs_return (default 50%)
- Extreme min/max daily returns and their dates

Usage:
    python src/audit_data.py --config configs/protocol.yaml
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def audit_stock(csv_path: Path, symbol: str, name: str, cfg: dict) -> tuple[dict, dict]:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    n_rows = len(df)
    date_min = df["Date"].min().strftime("%Y-%m-%d")
    date_max = df["Date"].max().strftime("%Y-%m-%d")

    zero_ohlc_mask = (
        (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Close"] <= 0)
        | df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    )
    n_zero_ohlc = int(zero_ohlc_mask.sum())

    n_zero_vol = int((df["Volume"] <= 0).sum()) if "Volume" in df.columns else 0

    ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    is_stale = (df[ohlcv] == df[ohlcv].shift(1)).all(axis=1)
    n_stale = int(is_stale.sum())

    # Flat days: next day's close identical to today's close
    valid_next = df["Close"].shift(-1).notna()
    pct_flat = float(((df["Close"].shift(-1) == df["Close"]) & valid_next).sum() / valid_next.sum() * 100.0)

    # Calendar gaps >= threshold
    gap_thresh = cfg["cleaning"]["break_min_calendar_gap_days"]
    cal_gap = df["Date"].diff().dt.days
    gap_mask = cal_gap >= gap_thresh
    n_gaps = int(gap_mask.sum())
    gap_events = [
        {"date": d.strftime("%Y-%m-%d"), "gap_days": int(g)}
        for d, g in zip(df.loc[gap_mask, "Date"], cal_gap.loc[gap_mask])
    ]

    # Single-day returns on positive closes
    pos_df = df.loc[~zero_ohlc_mask].copy().reset_index(drop=True)
    ret = pos_df["Close"].pct_change()
    jump_thresh = cfg["cleaning"]["break_min_abs_return"]
    jump_mask = ret.abs() >= jump_thresh
    n_jumps = int(jump_mask.sum())
    jump_events = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "return_pct": round(float(r * 100.0), 2),
            "prev_close": round(float(pc), 4),
            "close": round(float(c), 4),
        }
        for d, r, pc, c in zip(
            pos_df.loc[jump_mask, "Date"],
            ret.loc[jump_mask],
            pos_df["Close"].shift(1).loc[jump_mask],
            pos_df.loc[jump_mask, "Close"],
        )
    ]

    min_idx = ret.idxmin()
    max_idx = ret.idxmax()
    min_ret = float(ret.loc[min_idx] * 100.0)
    min_ret_date = pos_df.loc[min_idx, "Date"].strftime("%Y-%m-%d")
    max_ret = float(ret.loc[max_idx] * 100.0)
    max_ret_date = pos_df.loc[max_idx, "Date"].strftime("%Y-%m-%d")

    summary_row = {
        "symbol": symbol,
        "name": name,
        "n_rows": n_rows,
        "meets_2500": n_rows >= cfg["study"]["min_trading_days"],
        "date_min": date_min,
        "date_max": date_max,
        "n_zero_ohlc": n_zero_ohlc,
        "n_zero_vol": n_zero_vol,
        "n_stale_dup": n_stale,
        "pct_flat_days": round(pct_flat, 2),
        "n_gaps_ge_6d": n_gaps,
        "n_jumps_ge_50pct": n_jumps,
        "min_ret_pct": round(min_ret, 2),
        "min_ret_date": min_ret_date,
        "max_ret_pct": round(max_ret, 2),
        "max_ret_date": max_ret_date,
    }
    details = {
        "symbol": symbol,
        "gap_events": gap_events,
        "jump_events": jump_events,
    }
    return summary_row, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Pre-modeling audit of raw CSE CSVs.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--raw-dir", default="data/raw_cse", help="Folder containing raw CSVs")
    parser.add_argument("--out-dir", default="data/audit", help="Output folder for audit reports")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details_all = {}
    for item in cfg["tickers"]:
        symbol = item["symbol"]
        file_stem = item["file_stem"]
        csv_path = raw_dir / f"{file_stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing raw CSV for {symbol}: {csv_path}")
        row, det = audit_stock(csv_path, symbol, item["name"], cfg)
        rows.append(row)
        details_all[symbol] = det

    summary_df = pd.DataFrame(rows)
    out_csv = out_dir / "audit_summary.csv"
    out_json = out_dir / "audit_details.json"
    summary_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(details_all, f, indent=2)

    print(f"Saved audit summary -> {out_csv}")
    print(f"Saved detailed events -> {out_json}\n")
    print(summary_df.to_string(index=False))

    print("\n=== Jumps >= 50% Detected ===")
    any_jump = False
    for sym, det in details_all.items():
        if det["jump_events"]:
            any_jump = True
            for ev in det["jump_events"]:
                print(f"  {sym}: {ev['date']} | return={ev['return_pct']}% ({ev['prev_close']} -> {ev['close']})")
    if not any_jump:
        print("  None")

    print("\n=== Calendar Gaps >= 6 Days (Counts & Largest Gaps) ===")
    for sym, det in details_all.items():
        top_gaps = sorted(det["gap_events"], key=lambda x: x["gap_days"], reverse=True)[:3]
        gap_str = ", ".join(f"{g['date']} ({g['gap_days']}d)" for g in top_gaps)
        print(f"  {sym}: {len(det['gap_events'])} gaps | top: {gap_str}")


if __name__ == "__main__":
    main()
