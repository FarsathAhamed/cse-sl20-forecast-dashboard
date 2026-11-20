"""
Fetch daily historical OHLCV data for the 11 pre-registered S&P SL20 stocks
defined in configs/protocol.yaml using Yahoo Finance (<SYMBOL>-N0000.CM).

Usage:
    python src/fetch_cse_data.py --config configs/protocol.yaml
"""

import argparse
from pathlib import Path
import time
import yaml
import pandas as pd
import yfinance as yf


def fetch_all(config_path: str, out_dir: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start_date = cfg["study"]["start_date"]
    # yfinance end date is exclusive; add 1 day to include end_date
    end_date = (pd.Timestamp(cfg["study"]["end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = []
    for item in cfg["tickers"]:
        symbol = item["symbol"]
        file_stem = item["file_stem"]
        yf_ticker = f"{symbol}-N0000.CM"
        csv_file = out_path / f"{file_stem}.csv"

        print(f"Fetching {yf_ticker} ({item['name']}) [{start_date} to {cfg['study']['end_date']}]...")
        try:
            ticker_obj = yf.Ticker(yf_ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=False)
            if df.empty:
                print(f"  WARNING: No data returned for {yf_ticker}")
                summary.append((symbol, yf_ticker, 0, None, None, "EMPTY"))
                continue

            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
            # Keep clean standard columns
            keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"] if c in df.columns]
            df = df[keep_cols].sort_values("Date").reset_index(drop=True)
            df.to_csv(csv_file, index=False)

            print(f"  Saved {len(df)} rows -> {csv_file} ({df['Date'].min()} to {df['Date'].max()})")
            summary.append((symbol, yf_ticker, len(df), df["Date"].min(), df["Date"].max(), "OK"))
        except Exception as exc:
            print(f"  ERROR for {yf_ticker}: {exc}")
            summary.append((symbol, yf_ticker, 0, None, None, f"ERROR: {exc}"))

        time.sleep(0.5)

    print("\n=== Fetch Summary ===")
    res_df = pd.DataFrame(summary, columns=["Symbol", "YF_Ticker", "Rows", "Start", "End", "Status"])
    print(res_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch CSE S&P SL20 historical OHLCV CSVs.")
    parser.add_argument("--config", default="configs/protocol.yaml", help="Path to protocol.yaml")
    parser.add_argument("--out-dir", default="data/raw_cse", help="Output folder for raw CSVs")
    args = parser.parse_args()
    fetch_all(args.config, args.out_dir)


if __name__ == "__main__":
    main()
