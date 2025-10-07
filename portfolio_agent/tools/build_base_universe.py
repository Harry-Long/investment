"""
Utility script to construct a broad US common-stock seed list.

It downloads the Nasdaq Trader symbol directory, filters out ETFs/test
issues and preferred/share classes that typically cause data issues, then
writes the remaining tickers to a plain text file.
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request
from typing import Iterable, List

import pandas as pd


def _normalize_ticker(raw: str) -> str:
    """
    Normalise tickers for yfinance compatibility:
      - strip whitespace
      - drop NASDAQ-specific suffixes like '^', '$', '#'
      - convert '.' to '-' (e.g. BRK.B -> BRK-B)
    """
    ticker = str(raw).strip().upper()
    for ch in ("^", "$", "#", "~"):
        ticker = ticker.replace(ch, "")
    ticker = ticker.replace("/", "-")
    ticker = ticker.replace(".", "-")
    return ticker


def build_base_universe(
    include_exchanges: Iterable[str] | None = None,
    exclude_keywords: Iterable[str] | None = None,
    url: str = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt",
    source_file: str | None = None,
) -> List[str]:
    if source_file:
        with open(source_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    else:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep="|")
    df.columns = [str(c).strip() for c in df.columns]
    # drop footer row (File Creation Time:)
    df = df[df["Symbol"].ne("File Creation Time")]
    df = df[df["Financial Status"] != "D"]
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]
    if include_exchanges:
        include_set = {str(ex).strip().upper() for ex in include_exchanges}
        exchange_col = None
        for candidate in ("Exchange", "Listing Exchange", "Market Category"):
            if candidate in df.columns:
                exchange_col = candidate
                break
        if exchange_col:
            code_map = {
                "A": "NYSE MKT",
                "P": "NYSE ARCA",
                "Q": "NASDAQ",
                "N": "NYSE",
                "Z": "BATS",
                "V": "IEXG",
            }
            def normalize_exchange(value: object) -> str:
                val = str(value).strip().upper()
                if val in code_map:
                    return code_map[val]
                return val
            mask = df[exchange_col].map(lambda v: normalize_exchange(v) in include_set).fillna(False)
            df = df[mask]
        else:
            print("[warn] Exchange column not found; skipping exchange filter.")
    if exclude_keywords:
        pattern = "|".join(exclude_keywords)
        mask = ~df["Security Name"].str.contains(pattern, case=False, na=False)
        df = df[mask]
    symbol_col = "NASDAQ Symbol" if "NASDAQ Symbol" in df.columns else "Symbol"
    tickers = [_normalize_ticker(tk) for tk in df[symbol_col]]
    tickers = [tk for tk in tickers if tk and not tk.endswith("W") and not tk.endswith("WS")]
    tickers = sorted(set(tickers))
    return tickers


def write_universe(tickers: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for tk in tickers:
            fh.write(f"{tk}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build broad US common stock universe.")
    parser.add_argument("--out", default="data/universe/all_us_common.txt", help="Output txt file.")
    parser.add_argument(
        "--source-file",
        help="Local path to Nasdaq Trader symbol directory (nasdaqtraded.txt). If omitted, download from Nasdaq.",
    )
    parser.add_argument(
        "--exchanges",
        nargs="*",
        default=["NASDAQ", "NYSE", "NYSE MKT", "NYSE Arca"],
        help="Exchange codes to include from the NASDAQ Trader symbol file.",
    )
    parser.add_argument(
        "--exclude-keywords",
        nargs="*",
        default=["PREFERRED", "UNIT", "WARRANT", "NOTE"],
        help="Security Name keywords to exclude.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI utility
    args = parse_args()
    tickers = build_base_universe(
        include_exchanges=args.exchanges,
        exclude_keywords=args.exclude_keywords,
        source_file=args.source_file,
    )
    write_universe(tickers, args.out)
    print(f"Wrote {len(tickers)} tickers to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
