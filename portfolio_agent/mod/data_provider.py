# mod/data_provider.py
import pandas as pd
import numpy as np
from typing import Iterable, Optional

def get_prices_synthetic(tickers=('ASSET1','ASSET2'), start='2023-01-03', periods=252*2, seed=42):
    """Generate simple synthetic price series for quick pipeline testing."""
    np.random.seed(seed); tickers=list(tickers); n=len(tickers)
    dates=pd.date_range(start, periods=periods, freq='B')
    mu=np.array([0.10,0.07,0.04,0.02]); vol=np.array([0.22,0.18,0.10,0.05])
    mu=(mu[:n] if n<=4 else np.array([mu[i%4] for i in range(n)])); vol=(vol[:n] if n<=4 else np.array([vol[i%4] for i in range(n)]))
    mu/=252.0; vol/=np.sqrt(252.0)
    prices=pd.DataFrame(index=dates, columns=tickers, dtype=float); prices.iloc[0]=100.0
    for t in range(1,len(dates)):
        prices.iloc[t]=prices.iloc[t-1]*(1+mu+np.random.normal(0,vol,size=n))
    return prices

def get_prices_stooq(tickers: Iterable[str], start: str, end: Optional[str]):
    """Fetch prices from Stooq via pandas-datareader. US tickers usually need .US suffix (e.g., AAPL.US)."""
    from pandas_datareader import data as pdr
    frames={}
    for tk in tickers:
        try:
            df=pdr.DataReader(tk,'stooq',start=start,end=end)
            if df is not None and not df.empty and 'Close' in df.columns:
                s=df.sort_index()['Close'].rename(tk).dropna()
                if s.size>0: frames[tk]=s
        except Exception as e:
            print(f"[warn] {tk}: {e}")
    if not frames:
        raise RuntimeError("No tickers fetched from Stooq. Check suffixes like .US, or date range.")
    return pd.concat(frames.values(),axis=1).sort_index().ffill().dropna(how='all')
