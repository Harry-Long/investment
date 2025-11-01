# Portfolio Optimisation Agent

## Workflow
- Load `policy.yaml`, resolve the StockRover-exported universe (`data.stockrover_file`).
- Pull historical prices from Stooq (or synthetic series) and optional benchmark data.
- Apply an optional pre-selection step (`portfolio.selection`).
- Optimise portfolio weights via one of the supported models, then persist weights, return series, and QuantStats-style reports.
- Optionally run a light backtest (`portfolio.backtest`) and emit additional diagnostics (risk contributions, correlations, drawdowns).

## Module Map
- **`mod/policy.py` & `mod/universe.py`** – policy parsing plus universe resolution prioritising StockRover files, with legacy inline lists still supported.
- **`mod/data_provider.py`** – price/metadata loader with Stooq and synthetic support; hooks remain for FMP/Yahoo if desired.
- **`mod/optimizer.py`** – unified optimisation entry point exposing multiple models (mean-variance guardrail, Black-Litterman, risk parity, equal-weight baseline).
- **`mod/selector.py`** – optional pre-optimisation asset selection (e.g. top by 1y return or Sharpe).
- **`mod/perf_metrics.py`, `mod/risk_tools.py`** – shared performance and risk computations used across optimisation, reporting, and backtesting.
- **`mod/reporting_extras.py`, `mod/qs_wrapper.py`** – QuantStats integration plus CSV/plot/text exports.
- **`mod/backtest.py`** – in-sample or walk-forward validation of the optimised weights.
- **`run.py`** – orchestrator that stitches together policy, data, optimisation, and reporting.

## Optimisation Models
The refactored optimiser accepts a price DataFrame (`columns = tickers`) and the policy dictionary. It returns a dictionary containing:

```json
{
  "weights": { "TICKER": float, ... },
  "ann_return": float,
  "ann_vol": float,
  "sharpe": float,
  "max_drawdown": float,
  "model": "mean_variance | black_litterman | risk_parity | equal_weight",
  "objective": str | null,
  "details": { ... },           # model-specific metadata (solver, BL views, risk parity stats)
  "guardrails": { ... },        # only populated for mean-variance guardrail mode
  "leverage": float,
  "frequency": int,
  "risk_free_rate": float
}
```

Supported models:
- **`mean_variance`** – original Sharpe maximisation engine with optional guardrails on volatility, drawdown, minimum return, and Sharpe threshold.
- **`black_litterman`** – closed-form posterior combining equilibrium weights (from market caps) with optional absolute views and configurable tau/risk aversion.
- **`risk_parity`** – inverse-volatility heuristic with risk contribution diagnostics.
- **`equal_weight`** – simple baseline for benchmarking downstream analytics.

## Configuration Highlights (`policy.yaml`)
- `data.stockrover_file`: required path to the universe exported from StockRover (CSV or TXT). Inline lists (`data.universe`) remain a fallback.
- `portfolio.optimization.model`: choose the optimisation engine (`mean_variance`, `black_litterman`, `risk_parity`, `equal_weight`).
- `portfolio.optimization.guardrails`: thresholds used by the mean-variance solver.
- `portfolio.optimization.black_litterman`: optional `market_caps`, `absolute_views`, `tau`, and `risk_aversion`.
- `portfolio.optimization.weight_bounds`, `long_only`, `leverage`: portfolio constraints applied across models.
- `portfolio.selection`: optional pre-filter before optimisation.
- `portfolio.concentration`: top-k concentration enforcement (Buffett mode only).
- `portfolio.backtest`: enable walk-forward validation and configure rebalance cadence, train/test windows, and trading costs.
- `reporting.output_dir`: destination for QuantStats HTML, weights, returns, and diagnostics.

## Outputs
- `output/weights_initial*.csv`, `output/weights_optimized*.csv` – equal-weight vs optimised allocations.
- `output/returns_*.csv` – daily return series for each allocation.
- QuantStats HTML summaries for equal-weight, optimised, and combined views.
- Optional backtest JSON (walk-forward metrics), correlation matrices, price charts, and text summaries when `reporting.export_extras` is enabled.
