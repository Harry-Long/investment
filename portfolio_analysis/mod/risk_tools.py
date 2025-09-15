import numpy as np
import pandas as pd

def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    logret = np.log(prices).diff().dropna()
    return logret.apply(np.exp) - 1.0

def portfolio_returns(ret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = pd.Series(weights).reindex(ret.columns).fillna(0.0)
    return (ret * w).sum(axis=1)

def portfolio_nav(pf_ret: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1 + pf_ret).cumprod()

def annualized_cov(returns: pd.DataFrame, periods_per_year=252) -> pd.DataFrame:
    return returns.cov() * periods_per_year

def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).fillna(0.0).values
    v = float(np.sqrt(max(w @ cov.values @ w, 0.0)))
    return v

def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    wv = weights.reindex(cov.index).fillna(0.0)
    total_vol = portfolio_volatility(wv, cov)
    mrc = cov @ wv
    rc = wv * mrc / total_vol
    return rc

def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()

def var_es_hist(ret_series: pd.Series, alpha=0.95) -> dict:
    q = ret_series.quantile(1 - alpha)
    es = ret_series[ret_series <= q].mean()
    return {"VaR": float(q), "ES": float(es)}
