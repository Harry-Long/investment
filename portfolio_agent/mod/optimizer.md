# Optimiser Module Reference

The `optimizer.py` module exposes a single public entry point, `optimize_portfolio`, plus the underlying `SharpeGuardrailOptimizer` used for the mean-variance model with guardrails. The refactor centres the project on portfolio optimisation driven by a StockRover-provided universe file.

## `optimize_portfolio(prices: pd.DataFrame, policy: dict) -> dict`

### Inputs
- `prices`: price history with a column per ticker and a datetime index.
- `policy`: parsed `policy.yaml`. Configuration is read from `policy["portfolio"]["optimization"]` (falling back to `policy["portfolio"]` or the top-level dictionary for legacy support).

### Output
Dictionary containing:
```json
{
  "weights": { "TICKER": float, ... },
  "ann_return": float,
  "ann_vol": float,
  "sharpe": float,
  "max_drawdown": float,
  "model": "mean_variance | black_litterman | risk_parity | equal_weight",
  "objective": "max_sharpe | max_expected_utility | risk_parity | equal_weight" | null,
  "details": { ... },     // solver metadata, BL posterior stats, risk parity diagnostics
  "guardrails": { ... },  // populated only in mean-variance mode
  "leverage": float,
  "frequency": int,
  "risk_free_rate": float
}
```
The weights array always matches the column order of `prices`. Infeasible optimisation results raise `ValueError`.

### Supported models
- **`mean_variance`** (default): maximises Sharpe ratio via `SharpeGuardrailOptimizer`, respecting optional guardrails (`max_volatility`, `max_drawdown`, `min_return`, `min_sharpe`) plus leverage/weight bounds.
- **`black_litterman`**: closed-form Black-Litterman allocation using equilibrium weights (from `market_caps`) and optional absolute views. Config keys live under `portfolio.optimization.black_litterman` and include `market_caps`, `absolute_views`, `tau`, and `risk_aversion`.
- **`risk_parity`**: inverse-volatility heuristic with risk contribution diagnostics (see `details["risk_contribution"]`).
- **`equal_weight`**: 1/N baseline for benchmarking downstream reports.

### Key configuration fields (`policy.yaml`)
- `portfolio.optimization.model`: choose the model (`mean_variance`, `black_litterman`, `risk_parity`, `equal_weight`).
- `portfolio.optimization.frequency`: periods per year used for annualisation (default 252).
- `portfolio.optimization.risk_free_rate`: annual risk-free rate used for Sharpe calculations.
- `portfolio.optimization.weight_bounds`, `long_only`, `leverage`: shared constraints across all models.
- `portfolio.optimization.guardrails`: thresholds passed to the mean-variance optimiser.
- `portfolio.optimization.black_litterman`: optional `market_caps`, `absolute_views`, `tau`, `risk_aversion`.

### Guardrail optimiser (`SharpeGuardrailOptimizer`)
The class remains available for more advanced use cases (e.g., walk-forward re-optimisation). It operates on a returns DataFrame and the same configuration dictionary. Calling `optimize()` returns the weight vector, while `summarize()` reports annualised return/volatility, Sharpe, and max drawdown for any candidate weights.
