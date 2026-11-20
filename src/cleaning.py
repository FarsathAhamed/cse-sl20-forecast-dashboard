"""
Phase 3: Uniform Cleaning & Structural Break Protocol for all 11 S&P SL20 stocks.

Applies the exact same deterministic cleaning rules from configs/protocol.yaml:
1. Sort by ascending Date and deduplicate dates.
2. Drop zero / non-positive / NaN OHLC rows.
3. Drop exact stale/copied consecutive OHLCV rows.
4. Drop non-trading zero-volume flat rows (Volume <= 0 and Open == High == Low == Close)
   so Yahoo Finance forward-filled market closures & holidays are removed.
5. Detect structural breaks:
   - Calendar gap >= break_min_calendar_gap_days (default 6 days)
   - OR single-day |return| >= break_min_abs_return (default 50%)
6. Assign continuous regime_id and flag post-break purge windows (break day + next 30 rows).

Usage:
    python src/cleaning.py --config configs/protocol.yaml
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def clean_stock_df(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    n_raw = len(df)

    clean_cfg = cfg["cleaning"]

    # 1. Drop all-zero / non-positive / NaN OHLC rows
    n_zero_dropped = 0
    if clean_cfg.get("drop_zero_ohlc", True):
        valid_price = (
            (df["Open"] > 0)
            & (df["High"] > 0)
            & (df["Low"] > 0)
            & (df["Close"] > 0)
            & df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        )
        n_zero_dropped = int((~valid_price).sum())
        df = df.loc[valid_price].reset_index(drop=True)

    # 1b. Drop corrupted OHLC rows where High < Low
    n_bad_ohlc_dropped = 0
    if clean_cfg.get("drop_invalid_high_lt_low", True):
        valid_hl = df["High"] >= df["Low"]
        n_bad_ohlc_dropped = int((~valid_hl).sum())
        df = df.loc[valid_hl].reset_index(drop=True)

    # 2. Drop exact stale/copied consecutive OHLCV rows
    n_stale_dropped = 0
    if clean_cfg.get("drop_identical_prev_row", True):
        ohlcv = ["Open", "High", "Low", "Close", "Volume"]
        is_stale = (df[ohlcv] == df[ohlcv].shift(1)).all(axis=1)
        n_stale_dropped = int(is_stale.sum())
        df = df.loc[~is_stale].reset_index(drop=True)

    # 3. Drop non-trading zero-volume flat rows (Volume <= 0 & O == H == L == C)
    n_zerovol_flat_dropped = 0
    if clean_cfg.get("drop_zero_volume_flat", True):
        is_zerovol_flat = (
            (df["Volume"] <= 0)
            & np.isclose(df["Open"], df["High"])
            & np.isclose(df["High"], df["Low"])
            & np.isclose(df["Low"], df["Close"])
        )
        n_zerovol_flat_dropped = int(is_zerovol_flat.sum())
        df = df.loc[~is_zerovol_flat].reset_index(drop=True)

    # 4. Detect structural breaks AFTER removing non-trading rows
    cal_gap = df["Date"].diff().dt.days
    raw_ret = df["Close"].pct_change()

    gap_break = cal_gap >= clean_cfg["break_min_calendar_gap_days"]
    jump_break = raw_ret.abs() >= clean_cfg["break_min_abs_return"]
    is_break = (gap_break | jump_break).fillna(False)

    df["is_break"] = is_break
    df["break_reason"] = np.where(
        gap_break & jump_break,
        "gap+jump",
        np.where(gap_break, "gap", np.where(jump_break, "jump", "")),
    )
    df["regime_id"] = is_break.cumsum()

    # 5. Flag post-break purge window (break day + next 30 rows within each regime >= 1)
    purge_n = clean_cfg["post_break_purge_rows"]
    df["days_since_break"] = df.groupby("regime_id").cumcount()
    df["in_purge_window"] = (df["regime_id"] > 0) & (df["days_since_break"] <= purge_n)

    # Gate 3 sanity assertions
    assert df["Date"].is_monotonic_increasing, "Dates are not strictly increasing!"
    assert (df["Close"] > 0).all(), "Non-positive Close found after cleaning!"
    assert (df["High"] >= df["Low"]).all(), "High < Low found after cleaning!"

    n_clean = len(df)
    n_purged = int(df["in_purge_window"].sum())
    n_usable_post_purge = n_clean - n_purged

    meta = {
        "n_raw": n_raw,
        "n_zero_dropped": n_zero_dropped,
        "n_bad_ohlc_dropped": n_bad_ohlc_dropped,
        "n_stale_dropped": n_stale_dropped,
        "n_zerovol_flat_dropped": n_zerovol_flat_dropped,
        "n_after_drops": n_clean,
        "n_breaks": int(is_break.sum()),
        "n_gap_breaks": int(gap_break.sum()),
        "n_jump_breaks": int(jump_break.sum()),
        "n_purged_rows": n_purged,
        "pct_purged": round(n_purged / n_clean * 100.0, 2) if n_clean else 0.0,
        "n_usable_rows": n_usable_post_purge,
        "n_remaining_zero_vol": int((df["Volume"] <= 0).sum()),
        "break_events": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "reason": r,
                "gap_days": int(g) if pd.notna(g) else None,
                "ret_pct": round(float(ret * 100.0), 2) if pd.notna(ret) else None,
            }
            for d, r, g, ret in zip(
                df.loc[is_break, "Date"],
                df.loc[is_break, "break_reason"],
                cal_gap.loc[is_break],
                raw_ret.loc[is_break],
            )
        ],
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Verify uniform cleaning & breaks across all 11 stocks.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--raw-dir", default="data/raw_cse", help="Folder with raw CSVs")
    parser.add_argument("--out-dir", default="data/audit", help="Folder to save cleaning summary")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    events_all = {}
    for item in cfg["tickers"]:
        sym = item["symbol"]
        csv_path = raw_dir / f"{item['file_stem']}.csv"
        raw_df = pd.read_csv(csv_path)
        _, meta = clean_stock_df(raw_df, cfg)
        events_all[sym] = meta["break_events"]
        rows.append({k: v for k, v in meta.items() if k != "break_events"} | {"symbol": sym})

    cols = [
        "symbol", "n_raw", "n_zero_dropped", "n_bad_ohlc_dropped", "n_stale_dropped", "n_zerovol_flat_dropped",
        "n_after_drops", "n_breaks", "n_gap_breaks", "n_jump_breaks",
        "n_purged_rows", "pct_purged", "n_usable_rows", "n_remaining_zero_vol",
    ]
    summary_df = pd.DataFrame(rows)[cols]
    summary_df.to_csv(out_dir / "cleaning_summary.csv", index=False)
    with open(out_dir / "cleaning_breaks.json", "w", encoding="utf-8") as f:
        json.dump(events_all, f, indent=2)

    print("Gate 3 Verification Passed — Cleaning Summary:\n")
    print(summary_df.to_string(index=False))

    print("\n=== Detected Structural Breaks (Sample per Stock) ===")
    for sym, evs in events_all.items():
        ev_str = "; ".join(
            f"{e['date']} ({e['reason']}, gap={e['gap_days']}d, ret={e['ret_pct']}%)"
            for e in evs
        )
        print(f"  {sym} ({len(evs)}): {ev_str}")


if __name__ == "__main__":
    main()
