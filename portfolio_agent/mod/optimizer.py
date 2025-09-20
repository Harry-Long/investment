# mod/optimizer.py
from __future__ import annotations
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier, objective_functions

def optimize_portfolio(prices: pd.DataFrame, policy: dict) -> dict:
    """
    Run PyPortfolioOpt based on a policy dict.
    Supports objectives: max_sharpe, min_volatility, efficient_risk, efficient_return, max_quadratic_utility.
    Policy can be under policy['portfolio']['optimization'] or top-level policy['optimization'].
    Returns a dict with weights and summary metrics.
    """
    port = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
    opt_cfg = (port.get("optimization") or {}) if isinstance(port, dict) else {}
    objective = (opt_cfg.get("objective") or "max_sharpe").lower().strip()

    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S  = risk_models.sample_cov(prices, frequency=252)

    wb = opt_cfg.get("weight_bounds", [0.0, 0.35])
    if isinstance(wb, (list, tuple)) and len(wb) == 2:
        weight_bounds = (float(wb[0]), float(wb[1]))
    else:
        weight_bounds = (0.0, 1.0)

    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    l2_gamma = float(opt_cfg.get("l2_gamma", 0.001))
    if l2_gamma > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)

    rf = float(opt_cfg.get("risk_free_rate", 0.0))

    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=rf)
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "efficient_risk":
        tv = opt_cfg.get("target_volatility", None)
        if tv is None:
            raise ValueError("optimization.objective=efficient_risk requires target_volatility in policy.yaml")
        ef.efficient_risk(target_volatility=float(tv), risk_free_rate=rf)
    elif objective == "efficient_return":
        tr = opt_cfg.get("target_return", None)
        if tr is None:
            raise ValueError("optimization.objective=efficient_return requires target_return in policy.yaml")
        ef.efficient_return(target_return=float(tr), market_neutral=False)
    elif objective == "max_quadratic_utility":
        ra = float(opt_cfg.get("risk_aversion", 1.0))
        ef.max_quadratic_utility(risk_aversion=ra, market_neutral=False)
    else:
        raise ValueError(f"Unsupported optimization.objective: {objective}")

    weights = ef.clean_weights()
    ann_return, ann_vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=rf)

    return {
        "weights": weights,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "objective": objective,
        "rf": rf,
        "l2_gamma": l2_gamma,
        "weight_bounds": weight_bounds,
    }
