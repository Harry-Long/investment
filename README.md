# Portfolio Agent Project (Static Allocation)

## Setup
We recommend conda/mamba:

```bash
mamba install -c conda-forge pandas numpy matplotlib quantstats pypfopt pandas-datareader pyyaml
```

## Run
Use Stooq with YAML universe/file:
```bash
python run.py --source stooq --export-extras
```

Override tickers and mode from CLI:
```bash
python run.py --source stooq --tickers AAPL.US MSFT.US NVDA.US --mode naive --export-extras
```

Use synthetic data:
```bash
python run.py --source synthetic --export-extras
```

## Files
- `run.py` — main orchestrator
- `mod/data_provider.py` — Stooq and synthetic price loaders
- `mod/qs_wrapper.py` — QuantStats wrappers
- `mod/risk_tools.py` — returns, covariance, VaR/ES, nav
- `mod/reporting_extras.py` — CSV/plots/text report
- `mod/policy.py` — policy parsing and mode/dates/benchmark resolution
- `mod/universe.py` — universe resolution (CLI > file > YAML)
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
