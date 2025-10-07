# Portfolio Agent Project (Static Allocation)

## Setup
We recommend conda/mamba:

```bash
mamba install -c conda-forge pandas numpy matplotlib quantstats pypfopt pandas-datareader pyyaml
```

## Run
1. (Optional one-time) Build a seed list for the candidate pool. For a focused S&P 1500 universe run:

   ```bash
   python -m portfolio_agent.tools.build_sp1500_universe --out data/universe/sp1500.txt
   ```

   If you prefer the full US equity list, use the Nasdaq-based helper and point it at a cached text file:

   ```bash
   python -m portfolio_agent.tools.build_base_universe \
       --source-file data/universe/nasdaqtraded.txt \
       --out data/universe/all_us_common.txt
   ```

   Update `policy.yaml` so `candidate_pool.base_universe_file` references the chosen output (`sp1500.txt` by default now).

2. Generate the ~200 stock candidate pool (writes snapshots to `data/universe/`):

   ```bash
   python -m portfolio_agent.mod.stock_analysis --task build --policy policy.yaml
   ```

   Progress logs appear every few dozen tickers while metadata and fundamentals download. Cached copies reside under `data/cache/` (controlled via `data.cache.*` in `policy.yaml`) so subsequent runs reuse fresh data instead of redownloading.

3. Execute the portfolio runner with the same policy file:

   ```bash
   python portfolio_agent/run.py [optional/path/to/config.yaml]
   ```

If no path is supplied the default `policy.yaml` at the repo root is used. Switch to synthetic prices by setting `data.source: synthetic`. When using yfinance you can tune rate limits with `data.yfinance_chunk_size`, `data.yfinance_call_delay`, and the retry settings in the YAML to avoid hitting Yahoo throttling.

## Files
- `portfolio_agent/run.py` — main orchestrator
- `mod/data_provider.py` — flexible loader (Stooq, yfinance, synthetic) + metadata/fundamentals
- `mod/stock_analysis.py` — candidate pool builder with scoring & persistence
- `portfolio_agent/tools/build_sp1500_universe.py` — helper for S&P 500/400/600 combined list
- `portfolio_agent/tools/build_base_universe.py` — helper for downloading all US common stocks
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
