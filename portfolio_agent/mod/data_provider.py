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

    It relies on yfinance for fundamentals/metadata and Stooq for price history by
    default, but both sources can be toggled via policy config.
    """

    def __init__(self, policy_cfg: Dict[str, object]):
        self.policy_cfg = policy_cfg or {}
        data_cfg = self.policy_cfg.get("data", {}) if isinstance(self.policy_cfg.get("data"), dict) else {}
        self.source = str(data_cfg.get("source", "stooq")).lower()
        use_yf_cfg = data_cfg.get("use_yfinance")
        if use_yf_cfg is None:
            # default: disable yfinance when using stooq/synthetic, enable otherwise
            self.use_yfinance = bool(yf is not None and self.source == "yfinance")
        else:
            self.use_yfinance = bool(use_yf_cfg) and yf is not None
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
        Return metadata for the seed ticker list. Requires yfinance for fundamentals.
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
        if not self.use_yfinance:
            log.info("Metadata retrieval via yfinance disabled; using placeholder values.")
            meta = pd.DataFrame(
                {
                    "ticker": tickers,
                    "sector": ["Unknown"] * len(tickers),
                    "is_etf": [False] * len(tickers),
                    "is_adr": [False] * len(tickers),
                    "list_date": [None] * len(tickers),
                    "mcap": [np.nan] * len(tickers),
                    "price": [np.nan] * len(tickers),
                    "adv20": [np.nan] * len(tickers),
                }
            )
            self._meta_cache = meta
            return meta.copy()
        if yf is None:
            raise RuntimeError("yfinance is required for metadata retrieval.")
        meta_rows: List[Dict[str, object]] = []
        chunk_size = self.yf_chunk_size if self.use_yfinance else len(tickers)
        chunks = list(_chunked(tickers, chunk_size))
        for idx, chunk in enumerate(chunks, 1):
            cached = []
            to_fetch = []
            if self.cache_enabled:
                for ticker in chunk:
                    cached_row = self._load_metadata_row_from_cache(ticker, as_of)
                    if cached_row is not None:
                        cached.append(cached_row)
                    else:
                        to_fetch.append(ticker)
            else:
                to_fetch = chunk
            if cached:
                meta_rows.extend(cached)
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
                log.info("Metadata chunk %d/%d processed (fetched %d, cached %d)", idx, len(chunks), len(to_fetch), len(cached))
                if self.use_yfinance and self.yf_call_delay > 0:
                    time.sleep(self.yf_call_delay)
            else:
                log.info("Metadata chunk %d/%d served entirely from cache", idx, len(chunks))
        meta = pd.DataFrame(meta_rows)
        meta = meta.drop_duplicates(subset=["ticker"])
        self._meta_cache = meta
        self._save_metadata_cache(tickers, as_of, meta)
        return meta.copy()

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
        if not self.use_yfinance or yf is None:
            log.info("Fundamental retrieval via yfinance disabled; returning NaNs.")
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
                }
            ).set_index("ticker")
        tickers = list(tickers)
        rows = []
        chunks = list(_chunked(tickers, 200))
        for idx, chunk in enumerate(chunks, 1):
            batch = yf.Tickers(" ".join(chunk))
            for tk in chunk:
                if tk in self._fundamental_cache:
                    info = self._fundamental_cache[tk]
                else:
                    info = self._load_fundamental_cache(tk)
                    if info is None:
                        info = self._fetch_yf_info(batch, tk)
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
                }
                rows.append(row)
            log.info("Fundamentals chunk %d/%d processed", idx, len(chunks))
            if self.use_yfinance and self.yf_call_delay > 0:
                time.sleep(self.yf_call_delay)
        return pd.DataFrame(rows).set_index("ticker")

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
        try:
            df.to_pickle(path)
        except Exception as exc:
            log.debug("Failed to cache metadata to %s: %s", path, exc)

    def _metadata_row_path(self, ticker: str, as_of: dt.date) -> Path:
        filename = f"{ticker}_{as_of.isoformat()}.json"
        return self._cache_path("metadata_rows", filename)

    def _save_metadata_row_to_cache(self, ticker: str, as_of: dt.date, row: Dict[str, object]) -> None:
        if not self.cache_enabled:
            return
        path = self._metadata_row_path(ticker, as_of)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(row, fh)
        except Exception as exc:
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
        try:
            series.to_pickle(path)
        except Exception as exc:
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
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(info, fh)
        except Exception as exc:
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
