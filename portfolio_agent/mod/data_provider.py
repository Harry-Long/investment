# mod/data_provider.py
from __future__ import annotations
import os
import time
import json
import logging
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt
from typing import Iterable, Optional, Sequence, Dict, List, Iterator

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore
from pandas_datareader import data as pdr  # type: ignore

try:
    from .fmp_client import FmpClient  # type: ignore
except ImportError:  # pragma: no cover - fallback if module missing
    FmpClient = None  # type: ignore

def get_prices_synthetic(tickers=('ASSET1','ASSET2'), start='2023-01-03', periods=252*2, seed=42):
    """Generate simple synthetic price series for quick pipeline testing."""
    np.random.seed(seed); tickers=list(tickers); n=len(tickers)
    dates=pd.date_range(start, periods=periods, freq='B')
    mu=np.array([0.10,0.07,0.04,0.02]); vol=np.array([0.22,0.18,0.10,0.05])
    mu=(mu[:n] if n<=4 else np.array([mu[i%4] for i in range(n)])); vol=(vol[:n] if n<=4 else np.array([vol[i%4] for i in range(n)]))
    mu/=252.0; vol/=np.sqrt(252.0)
    prices=pd.DataFrame(index=dates, columns=tickers, dtype=float); prices.iloc[0]=100.0
    for t in range(1,len(dates)):
        prices.iloc[t]=prices.iloc[t-1]*(1+mu+np.random.normal(0,vol,size=n))
    return prices

def get_prices_stooq(tickers: Iterable[str], start: str, end: Optional[str]):
    """Fetch prices from Stooq via pandas-datareader. US tickers usually need .US suffix (e.g., AAPL.US)."""
    frames={}
    for tk in tickers:
        try:
            df=pdr.DataReader(tk,'stooq',start=start,end=end)
            if df is not None and not df.empty and 'Close' in df.columns:
                s=df.sort_index()['Close'].rename(tk).dropna()
                if s.size>0: frames[tk]=s
        except Exception as e:
            print(f"[warn] {tk}: {e}")
    if not frames:
        raise RuntimeError("No tickers fetched from Stooq. Check suffixes like .US, or date range.")
    return pd.concat(frames.values(),axis=1).sort_index().ffill().dropna(how='all')


log = logging.getLogger(__name__)


class DataProvider:
    """
    Candidate-pool friendly provider that wraps price, metadata, and fundamentals.

    Price history defaults to Stooq with metadata/fundamentals sourced from FMP
    (or yfinance when enabled). Behaviour is controlled via the policy config.
    """

    def __init__(self, policy_cfg: Dict[str, object]):
        self.policy_cfg = policy_cfg or {}
        data_cfg = self.policy_cfg.get("data", {}) if isinstance(self.policy_cfg.get("data"), dict) else {}
        raw_price_source = data_cfg.get("price_source", data_cfg.get("source", "stooq"))
        self.price_source = str(raw_price_source or "stooq").strip().lower()
        self.source = self.price_source  # backward compatibility for existing callers
        self.metadata_source = str(data_cfg.get("metadata_source", "yfinance") or "yfinance").strip().lower()
        self.fundamentals_source = str(data_cfg.get("fundamentals_source", self.metadata_source) or self.metadata_source).strip().lower()
        use_yf_cfg = data_cfg.get("use_yfinance")
        default_use_yf = any(src == "yfinance" for src in (self.price_source, self.metadata_source, self.fundamentals_source))
        if use_yf_cfg is None:
            self.use_yfinance = bool(yf is not None and default_use_yf)
        else:
            self.use_yfinance = bool(use_yf_cfg) and yf is not None
        if default_use_yf and not self.use_yfinance:
            log.warning("yfinance requested in configuration but disabled; Yahoo-dependent data will fall back to placeholders.")
        fmp_cfg = data_cfg.get("fmp", {}) if isinstance(data_cfg.get("fmp"), dict) else {}
        api_keys_cfg = data_cfg.get("api_keys", {}) if isinstance(data_cfg.get("api_keys"), dict) else {}
        raw_key = api_keys_cfg.get("fmp")
        self.fmp_api_key = str(raw_key).strip() if raw_key is not None else ""
        self.fmp_batch_size = int(fmp_cfg.get("batch_size", 100))
        self.fmp_call_delay = float(fmp_cfg.get("call_delay", 12.0))
        self.fmp_max_retries = int(fmp_cfg.get("max_retries", 3))
        self.fmp_timeout = float(fmp_cfg.get("timeout", 30.0))
        base_url_cfg = str(fmp_cfg.get("base_url", "https://financialmodelingprep.com/stable") or "").strip()
        self.fmp_base_url = self._normalize_fmp_base(base_url_cfg) if base_url_cfg else "https://financialmodelingprep.com/stable"
        self.fmp_error_retry_seconds = float(fmp_cfg.get("error_retry_seconds", 3600.0))
        self.fmp_adapt_rate = bool(fmp_cfg.get("adapt_rate", True))
        self.fmp_max_call_delay = fmp_cfg.get("max_call_delay")
        if self.fmp_max_call_delay is not None:
            try:
                self.fmp_max_call_delay = float(self.fmp_max_call_delay)
            except (TypeError, ValueError):
                self.fmp_max_call_delay = None
        self.fmp_success_threshold = int(fmp_cfg.get("success_threshold", 5))
        self.fmp_log_each = bool(fmp_cfg.get("log_each_ticker", fmp_cfg.get("log_each_symbol", True)))
        self._fmp_client: Optional[FmpClient] = None
        self.yf_chunk_size = int(data_cfg.get("yfinance_chunk_size", 50))
        self.yf_call_delay = float(data_cfg.get("yfinance_call_delay", 0.2))
        self.yf_max_retries = int(data_cfg.get("yfinance_max_retries", 3))
        self.yf_retry_backoff = float(data_cfg.get("yfinance_retry_backoff", 2.0))
        cache_cfg = data_cfg.get("cache", {}) if isinstance(data_cfg.get("cache"), dict) else {}
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        cache_dir = cache_cfg.get("dir", "data/cache")
        self.cache_dir = Path(cache_dir)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        expiry_days = cache_cfg.get("expiry_days")
        self.cache_expiry_seconds: Optional[float]
        try:
            if expiry_days is None:
                self.cache_expiry_seconds = 3 * 86400.0
            else:
                exp = float(expiry_days)
                self.cache_expiry_seconds = exp * 86400.0 if exp > 0 else None
        except Exception:
            self.cache_expiry_seconds = 3 * 86400.0
        cand_cfg = self.policy_cfg.get("candidate_pool", {}) if isinstance(self.policy_cfg.get("candidate_pool"), dict) else {}
        self.base_universe_file = cand_cfg.get("base_universe_file")
        self._meta_cache: Optional[pd.DataFrame] = None
        self._fundamental_cache: Dict[str, Dict[str, float]] = {}

    # ----------- public API -----------
    def list_us_equities(self, as_of: dt.date) -> pd.DataFrame:
        """
        Return metadata for the seed ticker list using the configured provider.
        """
        tickers = self._resolve_base_universe()
        if not tickers:
            raise RuntimeError("Candidate pool requires a base universe list. Set candidate_pool.base_universe_file or data.universe.")
        if self._meta_cache is not None:
            return self._meta_cache.copy()
        if self.cache_enabled:
            cached_meta = self._load_metadata_cache(tickers, as_of)
            if cached_meta is not None:
                log.info("Loaded metadata for %d tickers from disk cache", len(cached_meta))
                self._meta_cache = cached_meta
                return cached_meta.copy()
        source = self.metadata_source
        if source == "fmp":
            meta = self._list_us_equities_fmp(tickers, as_of)
        elif source == "yfinance" and self.use_yfinance and yf is not None:
            meta = self._list_us_equities_yf(tickers, as_of)
        elif source == "yfinance":
            log.warning("yfinance metadata requested but unavailable; returning placeholder values.")
            meta = self._metadata_placeholder(tickers)
        else:
            log.info("Metadata source '%s' not recognised; returning placeholder values.", source)
            meta = self._metadata_placeholder(tickers)
        self._meta_cache = meta
        if self.cache_enabled:
            self._save_metadata_cache(tickers, as_of, meta)
        return meta.copy()

    def _metadata_placeholder(self, tickers: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": list(tickers),
                "sector": ["Unknown"] * len(tickers),
                "is_etf": [False] * len(tickers),
                "is_adr": [False] * len(tickers),
                "list_date": [None] * len(tickers),
                "mcap": [np.nan] * len(tickers),
                "price": [np.nan] * len(tickers),
                "adv20": [np.nan] * len(tickers),
            }
        )

    def _list_us_equities_yf(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        if yf is None or not self.use_yfinance:
            return self._metadata_placeholder(tickers)
        meta_rows: List[Dict[str, object]] = []
        chunk_size = self.yf_chunk_size if self.use_yfinance else len(tickers)
        chunks = list(_chunked(tickers, chunk_size))
        for idx, chunk in enumerate(chunks, 1):
            cached_rows: List[Dict[str, object]] = []
            to_fetch: List[str] = []
            if self.cache_enabled:
                for ticker in chunk:
                    cached_row = self._load_metadata_row_from_cache(ticker, as_of)
                    if cached_row is not None:
                        cached_rows.append(cached_row)
                    else:
                        to_fetch.append(ticker)
            else:
                to_fetch = list(chunk)
            if cached_rows:
                meta_rows.extend(cached_rows)
            if to_fetch:
                batch = yf.Tickers(" ".join(to_fetch)) if self.use_yfinance else None
                for ticker in to_fetch:
                    info = self._fetch_yf_info(batch, ticker) if self.use_yfinance else {}
                    price = info.get("regularMarketPrice")
                    mcap = info.get("marketCap")
                    adv = info.get("averageDailyVolume10Day") or info.get("averageVolume")
                    sector = info.get("sector") or "Unknown"
                    is_etf = bool(info.get("quoteType") == "ETF")
                    is_adr = bool(info.get("quoteType") == "ADR")
                    list_raw = info.get("firstTradeDateEpochSeconds") or info.get("ipoExpectedDate")
                    list_date = None
                    if list_raw is not None:
                        try:
                            if isinstance(list_raw, (int, float)):
                                list_date = pd.to_datetime(list_raw, unit="s").date()
                            else:
                                list_date = pd.to_datetime(list_raw).date()
                        except Exception:
                            list_date = None
                    row = {
                        "ticker": ticker,
                        "sector": sector or "Unknown",
                        "is_etf": is_etf,
                        "is_adr": is_adr,
                        "list_date": list_date,
                        "mcap": float(mcap) if mcap else np.nan,
                        "price": float(price) if price else np.nan,
                        "adv20": float(adv) if adv else np.nan,
                    }
                    meta_rows.append(row)
                    self._save_metadata_row_to_cache(ticker, as_of, row)
                log.info(
                    "Metadata chunk %d/%d processed via yfinance (fetched %d, cached %d)",
                    idx,
                    len(chunks),
                    len(to_fetch),
                    len(cached_rows),
                )
                if self.use_yfinance and self.yf_call_delay > 0:
                    time.sleep(self.yf_call_delay)
            else:
                log.info("Metadata chunk %d/%d served entirely from cache", idx, len(chunks))
        meta = pd.DataFrame(meta_rows)
        if meta.empty:
            return self._metadata_placeholder(tickers)
        return meta.drop_duplicates(subset=["ticker"])

    def _list_us_equities_fmp(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        client = self._get_fmp_client()
        meta_rows: List[Dict[str, object]] = []
        chunks = list(_chunked(tickers, self.fmp_batch_size if self.fmp_batch_size > 0 else len(tickers)))
        for idx, chunk in enumerate(chunks, 1):
            cached_rows: List[Dict[str, object]] = []
            to_fetch: List[str] = []
            if self.cache_enabled:
                for ticker in chunk:
                    cached_row = self._load_metadata_row_from_cache(ticker, as_of)
                    if cached_row is not None:
                        cached_rows.append(cached_row)
                    else:
                        to_fetch.append(ticker)
            else:
                to_fetch = list(chunk)
            if cached_rows:
                meta_rows.extend(cached_rows)
            fetched_count = 0
            if to_fetch:
                profiles = client.fetch_profiles(to_fetch)
                for ticker in to_fetch:
                    profile = profiles.get(ticker, {})
                    row = self._build_metadata_row_from_profile(ticker, profile)
                    meta_rows.append(row)
                    self._save_metadata_row_to_cache(ticker, as_of, row)
                fetched_count = len(to_fetch)
            log.info(
                "Metadata chunk %d/%d processed via FMP (fetched %d, cached %d)",
                idx,
                len(chunks),
                fetched_count,
                len(cached_rows),
            )
        meta = pd.DataFrame(meta_rows)
        if meta.empty:
            return self._metadata_placeholder(tickers)
        meta = meta.drop_duplicates(subset=["ticker"])
        present = set(meta["ticker"].tolist())
        missing = [tk for tk in tickers if tk not in present]
        if missing:
            filler = self._metadata_placeholder(missing)
            meta = pd.concat([meta, filler], ignore_index=True)
        return meta

    def _build_metadata_row_from_profile(self, ticker: str, profile: Dict[str, object]) -> Dict[str, object]:
        profile = profile or {}
        sector = str(profile.get("sector", "Unknown") or "Unknown")
        is_etf = self._coerce_bool(profile.get("isEtf"))
        is_adr = self._coerce_bool(profile.get("isAdr"))
        list_date = self._parse_date(profile.get("ipoDate"))
        price = self._safe_float(profile.get("price"))
        adv = self._safe_float(
            profile.get("volAvg")
            or profile.get("volavg")
            or profile.get("avgVolume")
            or profile.get("averageVolume")
        )
        mcap = self._safe_float(profile.get("mktCap") or profile.get("marketCap"))
        row = {
            "ticker": ticker,
            "sector": sector or "Unknown",
            "is_etf": is_etf,
            "is_adr": is_adr,
            "list_date": list_date.isoformat() if isinstance(list_date, dt.date) else None,
            "mcap": mcap,
            "price": price,
            "adv20": adv,
        }
        # Include optional descriptive columns when available
        if profile:
            name = profile.get("companyName") or profile.get("company_name")
            if name:
                row["name"] = str(name)
            exchange = profile.get("exchangeShortName") or profile.get("exchange")
            if exchange:
                row["exchange"] = str(exchange)
        return row

    def get_history(self, tickers: Sequence[str], end: dt.date, lookback_days: int, field: str = "close") -> pd.DataFrame:
        tickers = list(tickers)
        start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        if self.source == "synthetic":
            prices = get_prices_synthetic(tuple(tickers), start=start, periods=lookback_days)
            if field.lower() == "close":
                return prices
            raise ValueError("Synthetic provider only supports close prices.")
        if self.source == "stooq":
            mapping = {tk: _stooq_symbol(tk) for tk in tickers}
            frame = get_prices_stooq(tuple(mapping.values()), start=start, end=end.strftime("%Y-%m-%d"))
            frame = frame.rename(columns={v: k for k, v in mapping.items() if v in frame.columns})
            if field.lower() == "close":
                return frame
            if field.lower() == "volume":
                frames = {}
                for tk, stooq_tk in mapping.items():
                    try:
                        df = pdr.DataReader(stooq_tk, "stooq", start=start, end=end.strftime("%Y-%m-%d"))
                        if df is not None and not df.empty and "Volume" in df.columns:
                            frames[tk] = df.sort_index()["Volume"].rename(tk)
                    except Exception:
                        continue
                if not frames:
                    return pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
                return pd.concat(frames.values(), axis=1).reindex(frame.index)
        # default to yfinance
        if yf is None:
            raise RuntimeError("yfinance required for price history when source != stooq.")
        frames = []
        cached_columns: Dict[str, pd.Series] = {}
        missing: List[str] = []
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        for tk in tickers:
            series = self._load_price_cache(tk, field.lower(), start_ts, end_ts)
            if series is not None:
                cached_columns[tk] = series
            else:
                missing.append(tk)
        if cached_columns:
            frames.append(pd.concat(cached_columns.values(), axis=1, keys=cached_columns.keys()))
        if missing:
            chunk_size = self.yf_chunk_size if self.use_yfinance else 100
            for idx, chunk in enumerate(_chunked(missing, chunk_size), 1):
                data = self._yf_download(symbols=chunk, start=start, end=end.strftime("%Y-%m-%d"), field=field)
                if data.empty:
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    panels = {}
                    for tk in chunk:
                        try:
                            series = data[tk][field.title()].rename(tk)
                        except KeyError:
                            continue
                        panels[tk] = series
                        self._save_price_cache(tk, field.lower(), series)
                    if panels:
                        frames.append(pd.concat(panels.values(), axis=1))
                else:
                    col = field.title()
                    if col in data.columns and len(chunk) == 1:
                        series = data[col].rename(chunk[0])
                        frames.append(series.to_frame())
                        self._save_price_cache(chunk[0], field.lower(), series)
                log.info("Price chunk %d fetched (missing batch) for field=%s", idx, field)
        if not frames:
            return pd.DataFrame(index=pd.Index([], dtype="datetime64[ns]"))
        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined

    def get_fundamentals(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        tickers = list(tickers)
        if not tickers:
            return pd.DataFrame(
                columns=[
                    "pe",
                    "pb",
                    "ev_to_ebitda",
                    "roe",
                    "gross_margin",
                    "profit_volatility",
                    "debt_to_equity",
                    "fcf_yield",
                    "roic",
                    "net_debt_to_ebitda",
                ]
            )
        source = self.fundamentals_source
        if source == "fmp":
            return self._get_fundamentals_fmp(tickers, as_of)
        if source == "yfinance" and self.use_yfinance and yf is not None:
            return self._get_fundamentals_yf(tickers, as_of)
        if source == "yfinance":
            log.warning("yfinance fundamentals requested but unavailable; returning NaNs.")
        else:
            log.info("Fundamental source '%s' not recognised; returning NaNs.", source)
        return self._fundamentals_placeholder(tickers)

    def _fundamentals_placeholder(self, tickers: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ticker": list(tickers),
                "pe": [np.nan] * len(tickers),
                "pb": [np.nan] * len(tickers),
                "ev_to_ebitda": [np.nan] * len(tickers),
                "roe": [np.nan] * len(tickers),
                "gross_margin": [np.nan] * len(tickers),
                "profit_volatility": [np.nan] * len(tickers),
                "debt_to_equity": [np.nan] * len(tickers),
                "fcf_yield": [np.nan] * len(tickers),
                "roic": [np.nan] * len(tickers),
                "net_debt_to_ebitda": [np.nan] * len(tickers),
            }
        ).set_index("ticker")

    def _get_fundamentals_yf(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        chunks = list(_chunked(tickers, self.yf_chunk_size if self.yf_chunk_size > 0 else len(tickers)))
        for idx, chunk in enumerate(chunks, 1):
            batch = yf.Tickers(" ".join(chunk)) if self.use_yfinance else None
            for tk in chunk:
                if tk in self._fundamental_cache:
                    info = self._fundamental_cache[tk]
                else:
                    info = self._load_fundamental_cache(tk)
                    if info is None:
                        info = self._fetch_yf_info(batch, tk) if self.use_yfinance else {}
                        if info:
                            self._save_fundamental_cache(tk, info)
                    self._fundamental_cache[tk] = info or {}
                row = {
                    "ticker": tk,
                    "pe": info.get("trailingPE"),
                    "pb": info.get("priceToBook"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                    "roe": info.get("returnOnEquity"),
                    "gross_margin": info.get("grossMargins"),
                    "profit_volatility": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "fcf_yield": None,
                    "roic": info.get("returnOnEquity"),
                    "net_debt_to_ebitda": None,
                }
                rows.append(row)
            log.info("Fundamentals chunk %d/%d processed via yfinance", idx, len(chunks))
            if self.use_yfinance and self.yf_call_delay > 0:
                time.sleep(self.yf_call_delay)
        return pd.DataFrame(rows).set_index("ticker")

    def _get_fundamentals_fmp(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        client = self._get_fmp_client()
        data_map: Dict[str, Dict[str, object]] = {}
        chunks = list(_chunked(tickers, self.fmp_batch_size if self.fmp_batch_size > 0 else len(tickers)))
        required_keys = {"returnOnInvestedCapitalTTM", "freeCashFlowYieldTTM", "netDebtToEBITDATTM"}
        for idx, chunk in enumerate(chunks, 1):
            to_fetch: List[str] = []
            for tk in chunk:
                info = self._fundamental_cache.get(tk)
                if info is None and self.cache_enabled:
                    info = self._load_fundamental_cache(tk)
                if info is not None and required_keys.issubset(info.keys()):
                    data_map[tk] = info
                    self._fundamental_cache[tk] = info
                else:
                    to_fetch.append(tk)
            if to_fetch:
                ratios = client.fetch_ratios(to_fetch)
                key_metrics = client.fetch_key_metrics(to_fetch)
                for tk in to_fetch:
                    payload = {}
                    payload.update(ratios.get(tk) or {})
                    payload.update(key_metrics.get(tk) or {})
                    for field in required_keys:
                        payload.setdefault(field, None)
                    info = payload
                    data_map[tk] = info
                    self._fundamental_cache[tk] = info
                    self._save_fundamental_cache(tk, info)
            log.info(
                "Fundamentals chunk %d/%d processed via FMP (fetched %d, cached %d)",
                idx,
                len(chunks),
                len(to_fetch),
                len(chunk) - len(to_fetch),
            )
        rows = []
        for tk in tickers:
            info = data_map.get(tk, {})
            row = {
                "ticker": tk,
                "pe": self._safe_float(info.get("peRatioTTM") or info.get("priceEarningsRatioTTM")),
                "pb": self._safe_float(info.get("priceToBookRatioTTM") or info.get("priceBookValueRatioTTM")),
                "ev_to_ebitda": self._safe_float(info.get("enterpriseToEbitdaTTM") or info.get("evToEbitdaTTM")),
                "roe": self._safe_float(info.get("returnOnEquityTTM")),
                "gross_margin": self._safe_float(info.get("grossProfitMarginTTM") or info.get("grossMarginTTM")),
                "profit_volatility": self._safe_float(info.get("netProfitMarginTTM") or info.get("profitMarginTTM")),
                "debt_to_equity": self._safe_float(info.get("debtEquityRatioTTM") or info.get("debtToEquityTTM")),
                "fcf_yield": self._safe_float(info.get("freeCashFlowYieldTTM")),
                "roic": self._safe_float(info.get("returnOnInvestedCapitalTTM") or info.get("returnOnEquityTTM")),
                "net_debt_to_ebitda": self._safe_float(info.get("netDebtToEBITDATTM")),
            }
            rows.append(row)
        df = pd.DataFrame(rows).set_index("ticker")
        return df

    def get_beta(self, tickers: Sequence[str], as_of: dt.date, lookback_days: int = 504, benchmark: Optional[str] = None) -> pd.Series:
        ref = benchmark or "SPY"
        tickers = list(tickers)
        prices = self.get_history(tickers + [ref], end=as_of, lookback_days=lookback_days, field="close")
        prices = prices.ffill().dropna()
        if ref not in prices.columns:
            return pd.Series(0.0, index=tickers)
        returns = prices.pct_change().dropna()
        bench = returns[ref]
        betas = {}
        for tk in tickers:
            if tk not in returns.columns:
                betas[tk] = 0.0
                continue
            cov = np.cov(returns[tk], bench)[0][1]
            var = np.var(bench)
            betas[tk] = cov / var if var != 0 else 0.0
        return pd.Series(betas)

    # ----------- internal helpers -----------
    def _get_fmp_client(self) -> FmpClient:
        if FmpClient is None:
            raise RuntimeError("FMP client module not available. Ensure portfolio_agent.mod.fmp_client is present.")
        if self._fmp_client is not None:
            return self._fmp_client
        key = str(self.fmp_api_key or "").strip()
        if not key or "REPLACE_WITH_FMP_KEY" in key.upper():
            env_key = os.environ.get("FMP_API_KEY", "")
            key = str(env_key or "").strip()
        if not key or "REPLACE_WITH_FMP_KEY" in key.upper():
            raise RuntimeError(
                "Financial Modeling Prep API key not configured. "
                "Update data.api_keys.fmp in the policy or set the FMP_API_KEY environment variable."
            )
        self.fmp_api_key = key
        self._fmp_client = FmpClient(
            api_key=key,
            base_url=self.fmp_base_url,
            batch_size=self.fmp_batch_size if self.fmp_batch_size > 0 else 100,
            call_delay=self.fmp_call_delay,
            max_retries=self.fmp_max_retries,
            timeout=self.fmp_timeout,
            error_retry_seconds=self.fmp_error_retry_seconds,
            adapt_rate=self.fmp_adapt_rate,
            max_call_delay=self.fmp_max_call_delay,
            success_threshold=self.fmp_success_threshold,
            log_each_symbol=self.fmp_log_each,
        )
        return self._fmp_client

    @staticmethod
    def _normalize_fmp_base(url: str) -> str:
        base = (url or "").strip()
        if not base:
            return "https://financialmodelingprep.com/stable"
        base = base.rstrip("/")
        if base.endswith("/stable"):
            return base
        for suffix in ("/api/v4", "/api/v3", "/api", "/v4", "/v3"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        base = base.rstrip("/")
        if not base:
            base = "https://financialmodelingprep.com"
        return f"{base}/stable"

    @staticmethod
    def _safe_float(value: object) -> float:
        if value is None:
            return np.nan
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        try:
            text = str(value).strip()
            if not text:
                return np.nan
            text = text.replace(",", "")
            return float(text)
        except Exception:
            return np.nan

    @staticmethod
    def _coerce_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        return text in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _parse_date(value: object) -> Optional[dt.date]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                # Treat as timestamp in seconds
                return pd.to_datetime(value, unit="s").date()
            return pd.to_datetime(value).date()
        except Exception:
            return None

    def _cache_path(self, category: str, filename: str) -> Path:
        path = self.cache_dir / category
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def _cache_is_fresh(self, path: Path) -> bool:
        if not self.cache_enabled:
            return False
        if self.cache_expiry_seconds is None:
            return True
        try:
            age = time.time() - path.stat().st_mtime
            return age <= self.cache_expiry_seconds
        except FileNotFoundError:
            return False

    def _load_metadata_cache(self, tickers: Sequence[str], as_of: dt.date) -> Optional[pd.DataFrame]:
        if not self.cache_enabled:
            return None
        key = _hash_ticker_set(tickers)
        filename = f"{as_of.isoformat()}_{key}.pkl"
        path = self._cache_path("metadata", filename)
        if not path.exists() or not self._cache_is_fresh(path):
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None

    def _save_metadata_cache(self, tickers: Sequence[str], as_of: dt.date, df: pd.DataFrame) -> None:
        if not self.cache_enabled:
            return
        key = _hash_ticker_set(tickers)
        filename = f"{as_of.isoformat()}_{key}.pkl"
        path = self._cache_path("metadata", filename)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            df.to_pickle(tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            log.debug("Failed to cache metadata to %s: %s", path, exc)

    def _metadata_row_path(self, ticker: str, as_of: dt.date) -> Path:
        filename = f"{ticker}_{as_of.isoformat()}.json"
        return self._cache_path("metadata_rows", filename)

    def _save_metadata_row_to_cache(self, ticker: str, as_of: dt.date, row: Dict[str, object]) -> None:
        if not self.cache_enabled:
            return
        path = self._metadata_row_path(ticker, as_of)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(row, fh)
            os.replace(tmp_path, path)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            log.debug("Failed to cache metadata row for %s: %s", ticker, exc)

    def _load_metadata_row_from_cache(self, ticker: str, as_of: dt.date) -> Optional[Dict[str, object]]:
        if not self.cache_enabled:
            return None
        path = self._metadata_row_path(ticker, as_of)
        if not path.exists() or not self._cache_is_fresh(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _load_price_cache(self, ticker: str, field: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
        if not self.cache_enabled:
            return None
        path = self._cache_path("prices", f"{field}_{ticker}.pkl")
        if not path.exists() or not self._cache_is_fresh(path):
            return None
        try:
            series = pd.read_pickle(path)
        except Exception:
            return None
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return None
            series = series.iloc[:, 0]
        if not isinstance(series, pd.Series) or series.empty:
            return None
        series = pd.Series(series).dropna().sort_index()
        if series.empty or series.index.min() > start or series.index.max() < end:
            return None
        return series.loc[start:end]

    def _save_price_cache(self, ticker: str, field: str, series: pd.Series) -> None:
        if not self.cache_enabled:
            return
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        series = series.dropna().sort_index()
        if series.empty:
            return
        path = self._cache_path("prices", f"{field}_{ticker}.pkl")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            series.to_pickle(tmp_path)
            os.replace(tmp_path, path)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            log.debug("Failed to cache price for %s (%s): %s", ticker, field, exc)

    def _load_fundamental_cache(self, ticker: str) -> Optional[Dict[str, float]]:
        if not self.cache_enabled:
            return None
        path = self._cache_path("fundamentals", f"{ticker}.json")
        if not path.exists() or not self._cache_is_fresh(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _save_fundamental_cache(self, ticker: str, info: Dict[str, float]) -> None:
        if not self.cache_enabled or not info:
            return
        path = self._cache_path("fundamentals", f"{ticker}.json")
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(info, fh)
            os.replace(tmp_path, path)
        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            log.debug("Failed to cache fundamentals for %s: %s", ticker, exc)

    def _fetch_yf_info(self, batch, ticker: str) -> Dict[str, float]:
        if not self.use_yfinance or batch is None:
            return {}
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.yf_max_retries + 1):
            try:
                result = batch.tickers[ticker].info  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
                if result is None:
                    return {}
                # Convert other types (e.g., pandas Series) to dict if possible
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return dict(result)
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < self.yf_max_retries:
                    delay = self.yf_call_delay * (self.yf_retry_backoff ** (attempt - 1))
                    delay = max(delay, 0.1)
                    log.warning(
                        "yfinance rate limit for %s (attempt %d/%d); sleeping %.2fs",
                        ticker,
                        attempt,
                        self.yf_max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                log.debug("yfinance info fetch failed for %s: %s", ticker, exc)
                return {}
        if last_exc:
            log.debug("yfinance info fetch giving up for %s: %s", ticker, last_exc)
        return {}

    def _yf_download(self, symbols: Sequence[str], start: str, end: str, field: str) -> pd.DataFrame:
        if not self.use_yfinance or yf is None:
            return pd.DataFrame()
        symbols_tuple = tuple(symbols)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.yf_max_retries + 1):
            try:
                data = yf.download(
                    symbols_tuple,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    group_by="ticker",
                )
                return data if data is not None else pd.DataFrame()
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < self.yf_max_retries:
                    delay = self.yf_call_delay * (self.yf_retry_backoff ** (attempt - 1))
                    delay = max(delay, 0.1)
                    log.warning(
                        "yfinance download rate limited for %s (attempt %d/%d); sleeping %.2fs",
                        ",".join(symbols_tuple),
                        attempt,
                        self.yf_max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                log.debug("yfinance download failed for %s: %s", symbols_tuple, exc)
                return pd.DataFrame()
        if last_exc:
            log.debug("yfinance download giving up for %s: %s", symbols_tuple, last_exc)
        return pd.DataFrame()

    def _resolve_base_universe(self) -> List[str]:
        if hasattr(self, "_base_universe"):
            return getattr(self, "_base_universe")
        tickers: List[str] = []
        if isinstance(self.base_universe_file, str) and self.base_universe_file.strip():
            tickers = self._load_universe_file(self.base_universe_file.strip())
        if not tickers:
            data_cfg = self.policy_cfg.get("data", {}) if isinstance(self.policy_cfg.get("data"), dict) else {}
            universe_file = data_cfg.get("universe_file")
            if isinstance(universe_file, str) and universe_file.strip():
                tickers = self._load_universe_file(universe_file.strip())
            elif isinstance(data_cfg.get("universe"), list):
                tickers = [str(t).strip().upper() for t in data_cfg["universe"] if t]
        setattr(self, "_base_universe", tickers)
        return tickers

    def _load_universe_file(self, path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        tickers: List[str] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                tickers.extend(parts)
        seen = set()
        uniq = []
        for tk in tickers:
            if tk not in seen:
                seen.add(tk)
                uniq.append(tk.upper())
        return uniq


def _chunked(seq: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _stooq_symbol(ticker: str) -> str:
    if ticker.endswith(".US"):
        return ticker
    if "." in ticker:
        return f"{ticker}.US"
    return f"{ticker}.US"


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ["rate limit", "too many requests", "429", "try again later"]
    return any(key in msg for key in keywords)


def _hash_ticker_set(tickers: Sequence[str]) -> str:
    joined = ",".join(sorted(tickers))
    return hashlib.md5(joined.encode()).hexdigest()
