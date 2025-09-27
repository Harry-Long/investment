from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, objective_functions, risk_models

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - surfaced during runtime if scipy is missing
    minimize = None

from .perf_metrics import annualize_return, equity_curve, max_drawdown, sharpe_ratio, volatility

TRADING_DAYS = 252


class MultiObjectiveOptimizer:
    """Weighted-sum multi-objective optimiser migrated from portfolio_optimizer."""

    def __init__(self, returns: pd.DataFrame, cfg: Dict):
        if minimize is None:
            raise ImportError(
                "scipy is required for engine=multi_objective. Install scipy or choose engine=pyportfolioopt."
            )

        if returns is None or returns.empty:
            raise ValueError("Multi-objective optimisation requires non-empty returns data")

        self.cfg = cfg or {}
        self.returns = returns.copy()
        self.tickers = list(self.returns.columns)
        self.n_assets = len(self.tickers)
        if self.n_assets == 0:
            raise ValueError("No assets available for optimisation")

        self.freq = int(self.cfg.get("frequency", TRADING_DAYS))

        leverage = self.cfg.get("leverage", 1.0)
        self.target_leverage = float(leverage if leverage not in (None, "") else 1.0)

        self.min_weight = float(self.cfg.get("min_weight", 0.0) or 0.0)
        self.max_weight = float(self.cfg.get("max_weight", 1.0) or 1.0)
        self.long_only = bool(self.cfg.get("long_only", True))

        self.objectives = self.cfg.get("objectives", {}) or {}
        self.targets = self.cfg.get("targets", {}) or {}
        backtest_cfg = self.cfg.get("backtest", {}) if isinstance(self.cfg.get("backtest"), dict) else {}
        self.risk_free_rate = float(
            self.cfg.get("risk_free_rate", backtest_cfg.get("risk_free_rate", 0.0)) or 0.0
        )

    def _bounds(self):
        lower = max(0.0, self.min_weight) if self.long_only else self.min_weight
        return tuple((lower, self.max_weight) for _ in range(self.n_assets))

    @staticmethod
    def _constraint_sum_to_target(weights: np.ndarray, target: float) -> float:
        return float(np.sum(weights) - target)

    def _constraints(self):
        return ({"type": "eq", "fun": lambda w: self._constraint_sum_to_target(w, self.target_leverage)},)

    def _prepare_returns(self, returns: Optional[pd.DataFrame]) -> pd.DataFrame:
        if returns is None:
            return self.returns.loc[:, self.tickers]
        data = returns.reindex(columns=self.tickers)
        return data.dropna(how="all")

    def _composite_loss(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        port_rets = returns.dot(weights).dropna()
        if port_rets.empty:
            return 1e6

        eq = equity_curve(port_rets)
        vol = volatility(port_rets, freq=self.freq)
        drawdown = abs(max_drawdown(eq))
        sharpe = sharpe_ratio(port_rets, risk_free_rate=self.risk_free_rate, freq=self.freq)
        ann_ret = annualize_return(port_rets, freq=self.freq)

        loss = 0.0
        obj = self.objectives

        if obj.get("maximize_sharpe"):
            term = sharpe if not np.isnan(sharpe) else -1e6
            loss += obj["maximize_sharpe"] * (-term)
        if obj.get("minimize_volatility"):
            term = vol if not np.isnan(vol) else 1e6
            loss += obj["minimize_volatility"] * term
        if obj.get("minimize_max_drawdown"):
            term = drawdown if not np.isnan(drawdown) else 1e6
            loss += obj["minimize_max_drawdown"] * term
        if obj.get("maximize_return"):
            term = ann_ret if not np.isnan(ann_ret) else -1e6
            loss += obj["maximize_return"] * (-term)

        tgt = self.targets
        tgt_vol = tgt.get("target_volatility")
        if tgt_vol is not None and not np.isnan(vol) and vol > float(tgt_vol):
            loss += 1000.0 * (vol - float(tgt_vol))

        tgt_dd = tgt.get("target_max_drawdown")
        if tgt_dd is not None and not np.isnan(drawdown) and drawdown > float(tgt_dd):
            loss += 1000.0 * (drawdown - float(tgt_dd))

        return float(loss)

    def optimize(self, returns_window: Optional[pd.DataFrame] = None) -> np.ndarray:
        data = self._prepare_returns(returns_window)
        if data.empty:
            raise ValueError("Returns window for optimisation is empty")

        x0 = np.full(self.n_assets, self.target_leverage / self.n_assets)
        res = minimize(
            fun=lambda w: self._composite_loss(w, data),
            x0=x0,
            bounds=self._bounds(),
            constraints=self._constraints(),
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
        )

        if not res.success or res.x is None:
            return x0
        return res.x

    def summarize(self, weights: np.ndarray, returns_window: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        data = self._prepare_returns(returns_window)
        port_rets = data.dot(weights).dropna()
        if port_rets.empty:
            return {
                "ann_return": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
            }

        eq = equity_curve(port_rets)
        return {
            "ann_return": float(annualize_return(port_rets, freq=self.freq)),
            "ann_vol": float(volatility(port_rets, freq=self.freq)),
            "sharpe": float(sharpe_ratio(port_rets, risk_free_rate=self.risk_free_rate, freq=self.freq)),
            "max_drawdown": float(max_drawdown(eq)),
        }


def optimize_portfolio(prices: pd.DataFrame, policy: dict) -> dict:
    """Run a portfolio optimisation based on policy configuration."""

    port = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
    opt_cfg = (port.get("optimization") or {}) if isinstance(port, dict) else {}

    engine = (opt_cfg.get("engine") or "pyportfolioopt").lower().strip()

    if engine == "multi_objective":
        returns = prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("Price history is insufficient to compute returns for optimisation")

        merged_cfg = dict(opt_cfg.get("multi_objective", {}))
        for key in (
            "objectives",
            "targets",
            "long_only",
            "min_weight",
            "max_weight",
            "leverage",
            "risk_free_rate",
            "backtest",
            "frequency",
        ):
            if key in opt_cfg and key not in merged_cfg:
                merged_cfg[key] = opt_cfg[key]

        if isinstance(port, dict):
            backtest_cfg = port.get("backtest")
            if isinstance(backtest_cfg, dict) and "backtest" not in merged_cfg:
                merged_cfg["backtest"] = backtest_cfg
                if "risk_free_rate" not in merged_cfg and "risk_free_rate" in backtest_cfg:
                    merged_cfg["risk_free_rate"] = backtest_cfg["risk_free_rate"]

        optimiser = MultiObjectiveOptimizer(returns=returns, cfg=merged_cfg)
        weights_arr = optimiser.optimize()
        summary = optimiser.summarize(weights_arr)
        weights = {ticker: float(w) for ticker, w in zip(returns.columns, weights_arr)}

        return {
            "weights": weights,
            "engine": engine,
            "objectives": optimiser.objectives,
            "targets": optimiser.targets,
            **summary,
        }

    objective = (opt_cfg.get("objective") or "max_sharpe").lower().strip()

    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    cov = risk_models.sample_cov(prices, frequency=TRADING_DAYS)

    wb = opt_cfg.get("weight_bounds", [0.0, 0.35])
    if isinstance(wb, (list, tuple)) and len(wb) == 2:
        weight_bounds = (float(wb[0]), float(wb[1]))
    else:
        weight_bounds = (0.0, 1.0)

    ef = EfficientFrontier(mu, cov, weight_bounds=weight_bounds)

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
        "engine": engine,
    }
