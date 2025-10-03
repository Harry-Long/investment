"""
Stock Analysis Module
=====================

This module provides a `StockAnalysis` class that scores individual
stocks based on four broad dimensions – fundamental, technical,
capital‑flow and news sentiment.  It can be used to replace or augment
the existing `selector` module in the portfolio agent project.

The high‑level workflow is:

1.  **Fundamental score:** Evaluate a company’s financial health using
    accounting ratios such as the price‑to‑earnings (P/E) and
    price‑to‑book (P/B) ratios.  These ratios measure whether a stock is
    trading at a fair value relative to its earnings and book value.
    According to Investopedia, fundamental analysts examine a company’s
    financial statements and broader economic indicators to determine
    intrinsic value【188779649735328†L344-L369】.  Stocks trading at a
    discount to their intrinsic value earn higher fundamental scores.
2.  **Technical score:** Analyse recent price behaviour to capture
    momentum and trend strength.  Technical indicators such as moving
    averages, the Relative Strength Index (RSI) and the Moving Average
    Convergence Divergence (MACD) are commonly used.  Technical
    indicators help identify entry/exit points and trend strength; they
    complement fundamental analysis and provide insight into market
    sentiment【524398118476234†L417-L425】.
3.  **Capital‑flow score:** Measure how capital flows into or out of a
    stock using volume‑weighted indicators like the Money Flow Index
    (MFI).  The MFI tracks both price and volume to confirm buying or
    selling pressure; readings above 80 suggest overbought conditions,
    while below 20 suggests oversold【884884277577294†L343-L361】.
4.  **News sentiment score:** Analyse the tone of recent news articles
    and social media posts related to the company.  Positive sentiment
    implies bullish outlook, while negative sentiment implies bearish.

This class is designed to be data‑source agnostic.  For example,
historical price data can come from the existing `data_provider` module
or directly from `yfinance`.  Fundamental data can come from Yahoo
Finance, a third‑party API or a CSV cache.  News sentiment requires an
external API (e.g. NewsAPI or a custom NLP model); here it is
implemented as a stub to be filled in later.

Example usage:

```python
from stock_analysis import StockAnalysis

# Analyse a list of tickers between 2020 and 2025
sa = StockAnalysis(tickers=["AAPL", "MSFT"], start_date="2020-01-01", end_date="2025-09-30")
scores = sa.analyse(weights={"fundamental":0.4, "technical":0.3, "capital":0.2, "news":0.1})
for ticker, score in scores.items():
    print(ticker, score)
```

Note: This module does not perform any I/O on its own.  If you wish to
integrate it with an existing `data_provider`, supply the price data
directly via the constructor.

"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except ImportError:
    # yfinance is not a strict dependency; price data can be passed in
    yf = None  # type: ignore


@dataclass
class StockAnalysis:
    """Analyse and score individual stocks across multiple dimensions.

    Parameters
    ----------
    tickers : Iterable[str]
        A list of ticker symbols to analyse.
    start_date : str
        Start date for price history (YYYY‑MM‑DD).
    end_date : str
        End date for price history (YYYY‑MM‑DD).
    price_data : Optional[Dict[str, pd.DataFrame]]
        Preloaded price data.  If provided, keys should be tickers and
        values should be DataFrames with columns `Open`, `High`, `Low`,
        `Close`, and `Volume` indexed by date.  If not provided and
        `yfinance` is available, price data will be downloaded.
    """

    tickers: Iterable[str]
    start_date: str
    end_date: str
    price_data: Optional[Dict[str, pd.DataFrame]] = None

    fundamental_scores: Dict[str, float] = field(default_factory=dict, init=False)
    technical_scores: Dict[str, float] = field(default_factory=dict, init=False)
    capital_scores: Dict[str, float] = field(default_factory=dict, init=False)
    news_scores: Dict[str, float] = field(default_factory=dict, init=False)

    def _get_price_data(self, ticker: str) -> pd.DataFrame:
        """Retrieve historical price data for a single ticker.

        If `self.price_data` is provided and contains the ticker,
        return it.  Otherwise, fetch using `yfinance` (if available).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with columns `Open`, `High`, `Low`,
            `Close`, and `Volume`.
        """
        if self.price_data and ticker in self.price_data:
            return self.price_data[ticker]
        if yf is None:
            raise RuntimeError(
                "Price data not provided and yfinance not available."
            )
        data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError(f"No price data found for {ticker}")
        return data

    def _get_fundamental_data(self, ticker: str) -> Dict[str, float]:
        """Retrieve fundamental metrics for a ticker.

        This method attempts to extract financial ratios such as the P/E and
        P/B ratios.  It first tries to use `yfinance` to fetch the
        information.  If that fails, it returns empty values.

        Returns
        -------
        Dict[str, float]
            Dictionary of fundamental metrics.
        """
        metrics: Dict[str, float] = {}
        if yf is not None:
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info  # type: ignore[attr-defined]
                # P/E ratio (ttm) and Price/Book ratio may be available
                pe = info.get("trailingPE") or info.get("forwardPE")
                pb = info.get("priceToBook")
                metrics["pe_ratio"] = pe if pe is not None else np.nan
                metrics["pb_ratio"] = pb if pb is not None else np.nan
                # Additional metrics could include return on equity, debt/equity, etc.
                roe = info.get("returnOnEquity")
                metrics["roe"] = roe if roe is not None else np.nan
            except Exception:
                pass
        return metrics

    @staticmethod
    def _compute_rsi(prices: pd.Series, window: int = 14) -> float:
        """Compute the Relative Strength Index (RSI).

        Parameters
        ----------
        prices : pd.Series
            Series of closing prices.
        window : int
            Look‑back window for RSI.

        Returns
        -------
        float
            Latest RSI value scaled to [0, 100].
        """
        delta = prices.diff().dropna()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(window).mean()
        roll_down = down.rolling(window).mean()
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.iloc[-1]

    @staticmethod
    def _compute_macd(prices: pd.Series) -> float:
        """Compute the MACD indicator value.

        MACD is calculated as the difference between the 12‑ and 26‑period
        exponential moving averages.  The signal line (9‑period EMA of the
        MACD) is not used here; instead we return the current MACD value.
        
        Returns
        -------
        float
        """
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd.iloc[-1]

    @staticmethod
    def _compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> float:
        """Compute the Money Flow Index (MFI).

        The MFI measures the flow of money into and out of an asset by
        analysing both price movements and trading volume.  High readings
        (above 80) indicate overbought conditions while low readings
        (below 20) indicate oversold【884884277577294†L343-L361】.
        
        Parameters
        ----------
        high, low, close : pd.Series
            Series of high, low and close prices.
        volume : pd.Series
            Series of traded volumes.
        window : int
            Look‑back window for MFI.

        Returns
        -------
        float
            Latest MFI value scaled to [0, 100].
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        # Positive and negative money flows
        delta_tp = typical_price.diff()
        positive_flow = np.where(delta_tp > 0, money_flow, 0.0)
        negative_flow = np.where(delta_tp < 0, money_flow, 0.0)
        # Rolling sums
        pos_sum = pd.Series(positive_flow).rolling(window).sum()
        neg_sum = pd.Series(negative_flow).rolling(window).sum()
        mfr = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return mfi.iloc[-1]

    def _score_fundamental(self, metrics: Dict[str, float]) -> float:
        """Compute a fundamental score from financial metrics.

        Lower P/E and P/B ratios are generally preferred because they may
        indicate undervaluation relative to earnings and book value.  Higher
        return on equity (ROE) is considered better.

        We invert ratios and normalise them to [0, 1] using simple
        heuristics.  Missing data results in a neutral score of 0.5.
        
        Returns
        -------
        float
            Fundamental score between 0 and 1.
        """
        pe = metrics.get("pe_ratio")
        pb = metrics.get("pb_ratio")
        roe = metrics.get("roe")
        # Invert P/E and P/B ratios: lower ratio → higher score
        def invert_ratio(value: Optional[float], scale: float = 40.0) -> float:
            if value is None or np.isnan(value) or value <= 0:
                return 0.5
            inv = 1.0 / (1.0 + value / scale)
            return max(0.0, min(inv, 1.0))
        pe_score = invert_ratio(pe, scale=40.0)
        pb_score = invert_ratio(pb, scale=10.0)
        roe_score = 0.5
        if roe is not None and not np.isnan(roe):
            # Higher ROE is better; cap at 1.0
            roe_score = min(max((roe + 1.0) / 2.0, 0.0), 1.0)
        # Weighted average of scores (equal weights)
        return np.mean([pe_score, pb_score, roe_score])

    def _score_technical(self, data: pd.DataFrame) -> float:
        """Compute a technical score from price data.

        The technical score combines trend and momentum indicators such as
        moving average crossovers, RSI and MACD.  A bullish configuration
        (short‑term MA above long‑term MA, RSI around 50 and positive
        MACD) yields higher scores.

        Returns
        -------
        float
            Technical score between 0 and 1.
        """
        close = data["Close"].dropna()
        ma_short = close.rolling(50).mean().iloc[-1]
        ma_long = close.rolling(200).mean().iloc[-1]
        ma_score = 0.5
        if not np.isnan(ma_short) and not np.isnan(ma_long):
            # Score is 1 if short MA is 5% above long MA, 0 if 5% below
            ratio = (ma_short - ma_long) / ma_long
            ma_score = (ratio + 0.05) / 0.10
            ma_score = min(max(ma_score, 0.0), 1.0)
        rsi_value = self._compute_rsi(close)
        # RSI around 50 is neutral; high RSI (70+) is considered overbought,
        # low RSI (30-) oversold.  Score is highest near 50.
        rsi_score = 1.0 - abs(rsi_value - 50.0) / 50.0
        macd_value = self._compute_macd(close)
        # Positive MACD is bullish; negative is bearish.  Scale between 0 and 1.
        macd_score = 0.5 + np.tanh(macd_value) / 2.0
        return np.mean([ma_score, rsi_score, macd_score])

    def _score_capital(self, data: pd.DataFrame) -> float:
        """Compute a capital‑flow score using MFI.

        A high MFI (80+) suggests overbought and yields a lower score; a
        low MFI (20-) suggests oversold and yields a higher score.  Values
        near 50 are neutral.【884884277577294†L343-L361】
        
        Returns
        -------
        float
            Capital‑flow score between 0 and 1.
        """
        mfi = self._compute_mfi(data["High"], data["Low"], data["Close"], data["Volume"])
        # Map MFI to score: 0 at 80+, 1 at 20-, linear in between
        if np.isnan(mfi):
            return 0.5
        if mfi >= 80.0:
            return 0.0
        if mfi <= 20.0:
            return 1.0
        return (80.0 - mfi) / 60.0

    def _score_news(self, ticker: str) -> float:
        """Placeholder for news sentiment analysis.

        News sentiment requires an external API (e.g. NewsAPI or an in‑house
        NLP model).  Here we return a neutral score of 0.5.  To implement
        sentiment, fetch recent headlines and compute sentiment polarity.
        
        Returns
        -------
        float
            News sentiment score between 0 and 1.
        """
        # TODO: integrate with a news sentiment API
        return 0.5

    def analyse(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Analyse all tickers and compute overall scores.

        Parameters
        ----------
        weights : dict, optional
            Weights for combining the four dimension scores.  Keys should
            include `fundamental`, `technical`, `capital`, and `news`.  If
            omitted, equal weights are used.

        Returns
        -------
        Dict[str, float]
            Mapping of ticker to overall score (0–1).  Higher scores
            indicate more attractive stocks.
        """
        if weights is None:
            weights = {
                "fundamental": 0.25,
                "technical": 0.25,
                "capital": 0.25,
                "news": 0.25,
            }
        # Normalise weights
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}
        scores: Dict[str, float] = {}
        for ticker in self.tickers:
            # Gather data
            price_df = self._get_price_data(ticker)
            metrics = self._get_fundamental_data(ticker)
            # Compute scores and cache them
            f_score = self._score_fundamental(metrics)
            t_score = self._score_technical(price_df)
            c_score = self._score_capital(price_df)
            n_score = self._score_news(ticker)
            self.fundamental_scores[ticker] = f_score
            self.technical_scores[ticker] = t_score
            self.capital_scores[ticker] = c_score
            self.news_scores[ticker] = n_score
            overall = (
                weights.get("fundamental", 0) * f_score
                + weights.get("technical", 0) * t_score
                + weights.get("capital", 0) * c_score
                + weights.get("news", 0) * n_score
            )
            scores[ticker] = overall
        return scores

    def rank(self, weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """Return tickers ordered by their overall scores (descending)."""
        scores = self.analyse(weights=weights)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
