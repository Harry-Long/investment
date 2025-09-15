# Portfolio analysis with QuantStats and PyPortfolioOpt (Stooq backend only)

This package is locked to Stooq for real data. Synthetic is available for offline testing.

## Install
mamba install -c conda-forge pandas-datareader quantstats pypfopt matplotlib pandas numpy

## Run
- Synthetic demo
  python run.py --source synthetic

- Real data from Stooq
  Note Stooq requires .US for US tickers.
  python run.py --source stooq --tickers AAPL.US MSFT.US --start 2018-01-01 --benchmark SPY.US --export-extras

Outputs go to output/ with filename suffixes:
- _synth for synthetic
- _real for stooq

Educational use only. Not investment advice.
