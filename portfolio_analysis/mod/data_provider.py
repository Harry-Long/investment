import pandas as pd
import numpy as np
from typing import Iterable, Optional

def _cycle_params(n: int):
    # Templates are cycled to match any ticker length
    mu_tpl = [0.10, 0.07, 0.04, 0.02]   # annualized returns
    vol_tpl = [0.22, 0.18, 0.10, 0.05]  # annualized vol
    mu = np.array([mu_tpl[i % len(mu_tpl)] for i in range(n)], dtype=float)
    vol = np.array([vol_tpl[i % len(vol_tpl)] for i in range(n)], dtype=float)
    return mu, vol

def get_prices_synthetic(tickers=('ASSET1','ASSET2'), start='2023-01-03', periods=252*2, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    tickers = list(tickers)
    n = len(tickers)
    dates = pd.date_range(start, periods=periods, freq='B')
    mu_ann, vol_ann = _cycle_params(n)
    mu_d = mu_ann / 252.0
    vol_d = vol_ann / (252.0 ** 0.5)
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    # uniform starting price
    prices.iloc[0] = 100.0
    for t in range(1, len(dates)):
        shock = np.random.normal(0, vol_d, size=n)
        prices.iloc[t] = prices.iloc[t-1] * (1 + mu_d + shock)
    return prices

def get_prices_stooq(tickers: Iterable[str], start: str, end: Optional[str]) -> pd.DataFrame:
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise RuntimeError("Please install pandas-datareader: mamba install -c conda-forge pandas-datareader") from e

    frames = {}
    for t in tickers:
        try:
            df = pdr.DataReader(t, 'stooq', start=start, end=end)
            if df is not None and not df.empty:
                df = df.sort_index()
                if 'Close' in df.columns:
                    s = df['Close'].rename(t).dropna()
                    if s.size > 0:
                        frames[t] = s
        except Exception as e:
            print(f"[warn] Failed to fetch {t} from Stooq: {e}")

    if not frames:
        raise RuntimeError("No tickers fetched from Stooq. Check symbols. US tickers usually need .US suffix.")

    prices = pd.concat(frames.values(), axis=1).sort_index().ffill().dropna(how='all')
    return prices
