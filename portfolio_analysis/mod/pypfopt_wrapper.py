import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions

def optimize_max_sharpe(prices: pd.DataFrame, risk_free_rate: float = 0.0, l2_gamma: float = 0.001) -> dict:
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.sample_cov(prices, frequency=252)
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    ann_return, ann_vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return {"weights": weights, "ann_return": float(ann_return), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
