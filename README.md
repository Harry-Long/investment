# Portfolio Agent – Optimisation Workflow

## Setup
- Create the provided conda environment (`environment.yml`) or install dependencies manually (`numpy`, `pandas`, `scipy`, `quantstats`, etc.).
- Copy your Financial Modeling Prep key into `policy.yaml` (`data.api_keys.fmp`) if you plan to pull fundamentals/metadata.
- Export the investable universe from StockRover (CSV or TXT) and point `data.stockrover_file` at the saved file.

## Running the pipeline
1. Adjust `policy.yaml` as needed:
   - Confirm `data.stockrover_file` references the latest StockRover export.
   - Choose an optimisation model under `portfolio.optimization.model` (`mean_variance`, `black_litterman`, `risk_parity`, or `equal_weight`).
   - (Optional) Provide a fixed-holdings CSV via `portfolio.optimization.fixed_positions_file` to keep specific tickers untouched; include a `ticker` column and optional `weight`/`lock` fields.
   - Supply optional guardrails, Black-Litterman views, or leverage/weight limits.
2. Execute the analysis:
   ```bash
   python portfolio_agent/run.py            # uses policy.yaml by default
   ```
   Provide an alternate policy path as the first argument to run multiple scenarios.
3. Collect results from the configured `reporting.output_dir` (defaults to `output/`): weights, return series, QuantStats HTML, and optional backtest diagnostics.

## Optimisation models at a glance
| Model | Description | Key policy knobs |
| ----- | ----------- | ---------------- |
| `mean_variance` | Sharpe maximisation with optional guardrails (volatility, drawdown, min return/ShARPE) | `portfolio.optimization.guardrails`, `weight_bounds`, `long_only`, `leverage` |
| `black_litterman` | Closed-form posterior using equilibrium weights and optional absolute views | `portfolio.optimization.black_litterman.*` (`market_caps`, `absolute_views`, `tau`, `risk_aversion`) |
| `risk_parity` | Inverse-volatility heuristic with risk contribution diagnostics | `weight_bounds`, `long_only`, `leverage` |
| `equal_weight` | Baseline 1/N allocation for benchmarking | none |

All models consume the same inputs (price history and policy) and return a dictionary containing weights, annualised return/volatility, Sharpe, max drawdown, model metadata, and leverage.

## Key files
- `policy.yaml` – single source of configuration (data sources, optimisation model, constraints, reporting).
- `portfolio_agent/run.py` – orchestration script.
- `portfolio_agent/mod/fixed_positions.py` – parser for locked positions supplied via CSV or inline policy.
- `portfolio_agent/mod/optimizer.py` – optimisation engines and result schema.
- `portfolio_agent/mod/universe.py` – StockRover universe loader with legacy fallbacks.
- `output/` – generated weights, reports, optional backtest artefacts.

## Tips
- Use `reporting.export_extras: true` to persist additional diagnostics (risk contributions, correlations, price charts, text summary).
- Enable `portfolio.backtest.enabled` for a quick walk-forward validation of the optimised weights.
- Switch `data.source: synthetic` to generate sandbox price paths when experimenting without live data.
- When supplying fixed holdings without explicit weights, the loader falls back to `portfolio.initial_weights` (or an equal slice of the universe) and notes that choice in the console output.
