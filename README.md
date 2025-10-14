# Portfolio Agent Project (Static Allocation)

## Setup
We recommend conda/mamba:

```bash
mamba install -c conda-forge pandas numpy matplotlib quantstats pypfopt pandas-datareader pyyaml requests
```

Before building the candidate pool with live data, supply a Financial Modeling Prep key via `policy.yaml` (`data.api_keys.fmp`) or export it as an environment variable:

```bash
export FMP_API_KEY="YOUR_KEY"
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

If no path is supplied the default `policy.yaml` at the repo root is used. Switch to synthetic prices by setting `data.source: synthetic`. When using yfinance you can tune rate limits with `data.yfinance_chunk_size`, `data.yfinance_call_delay`, and the retry settings in the YAML to avoid hitting Yahoo throttling. FMP pacing is controlled via `data.fmp.batch_size` and `data.fmp.call_delay` (default 12 s between calls for the free tier).

## Files
- `portfolio_agent/run.py` — main orchestrator
- `mod/data_provider.py` — flexible loader (Stooq, FMP, yfinance, synthetic) + metadata/fundamentals
- `mod/fmp_client.py` — thin Financial Modeling Prep batch client (profiles + ratios)
- `mod/stock_analysis.py` — candidate pool builder with scoring & persistence
- `portfolio_agent/tools/build_sp1500_universe.py` — helper for S&P 500/400/600 combined list
- `portfolio_agent/tools/build_base_universe.py` — helper for downloading all US common stocks

## Data Pipeline Overview

1. **Data Provider** (`portfolio_agent/mod/data_provider.py`) normalizes the policy settings, hydrates metadata via FMP profiles, merges ratios and key metrics, and persists everything to disk with atomic cache writes.
2. **FMP Client** (`portfolio_agent/mod/fmp_client.py`) drives the live requests, adapts the call delay when rate limits are hit, skips symbols that return premium-only errors for a cooldown period, and can optionally suppress per-ticker logging (`data.fmp.log_each_ticker: false`).
3. **Candidate Builder** (`portfolio_agent/mod/stock_analysis.py`) applies universe filters, computes factors with smarter fallbacks (PE/PB/EV, FCF yield, ROE/ROIC, leverage), neutralises exposures, and exports the selected universe snapshot.
4. **Portfolio Runner** (`portfolio_agent/run.py`) loads the policy, resolves the universe, pulls prices (Stooq or synthetic), performs selection, runs optimisations, and writes QuantStats reports.

### Key Configuration Knobs

- `data.fmp.adapt_rate` / `max_call_delay` / `success_threshold` tune the adaptive throttling.
- `data.fmp.error_retry_seconds` controls how long to skip tickers that hit premium or auth errors.
- `data.fmp.log_each_ticker` toggles per-ticker logging for quieter builds.
- `candidate_pool.base_universe_file` can be swapped for a pilot subset while caches warm up.
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
