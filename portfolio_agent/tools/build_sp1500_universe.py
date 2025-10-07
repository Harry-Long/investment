"""
Build a base universe using S&P 1500 constituents (S&P 500 + 400 + 600).

The script scrapes Wikipedia tables and saves tickers to a text file. It
normalises tickers for Yahoo Finance (replace '.' with '-'), deduplicates,
and sorts alphabetically.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Iterable, List
from io import StringIO

import pandas as pd

WIKI_SOURCES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def _normalize(symbol: str) -> str:
    ticker = str(symbol).strip().upper()
    return ticker.replace(".", "-").replace("/", "-")


def _fetch_table(url: str, cache_path: str | None = None) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached_html = fh.read()
        tables = pd.read_html(StringIO(cached_html), header=0)
        if tables:
            return tables[0]
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(html)
    tables = pd.read_html(StringIO(html), header=0)
    if not tables:
        raise RuntimeError(f"No tables found at {url}")
    return tables[0]


def build_sp1500_universe(include: Iterable[str] | None = None, cache_dir: str | None = None) -> List[str]:
    include = list(include or WIKI_SOURCES.keys())
    tickers: List[str] = []
    for key in include:
        url = WIKI_SOURCES.get(key.lower())
        if not url:
            raise ValueError(f"Unknown index key '{key}'. Expected one of {list(WIKI_SOURCES)}.")
        cache_path = (
            os.path.join(cache_dir, f"{key.lower()}.html") if cache_dir else None
        )
        table = _fetch_table(url, cache_path=cache_path)
        symbol_col = None
        for candidate in ("Symbol", "Ticker symbol", "Ticker"):
            if candidate in table.columns:
                symbol_col = candidate
                break
        if symbol_col is None:
            raise RuntimeError(f"Could not find ticker column in table from {url}")
        tickers.extend(table[symbol_col].dropna().tolist())
    norm = [_normalize(tk) for tk in tickers if tk]
    uniq = sorted(set(norm))
    return uniq


def write_universe(tickers: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for tk in tickers:
            fh.write(f"{tk}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S&P 1500 base universe.")
    parser.add_argument(
        "--include",
        nargs="*",
        default=list(WIKI_SOURCES.keys()),
        help="Subset of indices to include (sp500, sp400, sp600).",
    )
    parser.add_argument(
        "--out",
        default="data/universe/sp1500.txt",
        help="Output file for the combined universe.",
    )
    parser.add_argument(
        "--cache-dir",
        help="Optional directory to cache downloaded HTML tables.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = parse_args()
    tickers = build_sp1500_universe(args.include, cache_dir=args.cache_dir)
    write_universe(tickers, args.out)
    print(f"Wrote {len(tickers)} tickers to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
