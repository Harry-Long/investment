# Portfolio Agent Project (Static Allocation)

## Setup
We recommend conda/mamba:

```bash
mamba install -c conda-forge pandas numpy matplotlib quantstats pypfopt pandas-datareader pyyaml
```

## Run
1. Adjust `portfolio_agent/policy.yaml` as needed (data source, reporting, portfolio settings).
2. Execute the runner with the policy file:

```bash
python portfolio_agent/run.py [optional/path/to/config.yaml]
```

If no path is supplied the default `portfolio_agent/policy.yaml` is used. Switch to synthetic prices by setting `data.source: synthetic` in the YAML.

## Files
- `portfolio_agent/run.py` — main orchestrator
- `mod/data_provider.py` — Stooq and synthetic price loaders
- `mod/qs_wrapper.py` — QuantStats wrappers
- `mod/risk_tools.py` — returns, covariance, VaR/ES, nav
- `mod/reporting_extras.py` — CSV/plots/text report
- `mod/policy.py` — policy parsing and mode/dates/benchmark resolution
- `mod/universe.py` — universe resolution (policy file)
- `mod/selector.py` — pre-optimization selection (top by return / Sharpe)
- `mod/optimizer.py` — PyPortfolioOpt objective dispatch + multi-objective engine
- `mod/perf_metrics.py` — shared performance/risk helpers
- `mod/backtest.py` — in-sample & walk-forward backtesting wrapper
- `mod/concentration.py` — buffet top-k concentration rule
- `policy.yaml` — full policy template
- `universe.txt` — sample universe list

## Notes
- US tickers on Stooq usually need `.US` suffix (e.g., `AAPL.US`).
- QuantStats HTML and extra CSVs will be written to `./output` by default.
- Optional backtesting (in-sample or walk-forward) can be enabled in `policy.yaml` under `portfolio.backtest`.
