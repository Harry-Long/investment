"""
Lightweight Financial Modeling Prep client for batch metadata and ratio requests.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

log = logging.getLogger(__name__)


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    size = max(1, size)
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


class FmpClient:
    """
    Minimal wrapper around Financial Modeling Prep stable REST endpoints.

    The client walks tickers sequentially and enforces a simple rate limit
    (call_delay) to remain within the free-tier allowance of roughly 5 calls
    per minute.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://financialmodelingprep.com/stable",
        batch_size: int = 100,
        timeout: float = 30.0,
        call_delay: float = 12.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
        error_retry_seconds: float = 3600.0,
        adapt_rate: bool = True,
        max_call_delay: Optional[float] = None,
        success_threshold: int = 5,
        log_each_symbol: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("FMP API key must be provided.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") or "https://financialmodelingprep.com/stable"
        self.batch_size = max(1, int(batch_size))
        self.timeout = float(timeout)
        self.call_delay = max(0.0, float(call_delay))
        self.max_retries = max(1, int(max_retries))
        self.session = session or requests.Session()
        self._last_call = 0.0
        self._last_status_code: Optional[int] = None
        self.error_retry_seconds = max(0.0, float(error_retry_seconds))
        self._error_blocks: Dict[str, Dict[str, float]] = {}
        self._last_error_reason: Dict[Tuple[str, str], Tuple[int, str]] = {}
        self._adapt_rate = bool(adapt_rate)
        self._min_call_delay = self.call_delay
        upper_bound = max_call_delay if max_call_delay is not None else max(self.call_delay * 4.0, self.call_delay + 30.0)
        self._max_call_delay = max(self.call_delay, float(upper_bound))
        self._success_streak = 0
        self._success_threshold = max(1, int(success_threshold))
        self._log_each_symbol = bool(log_each_symbol)

    def fetch_profiles(self, tickers: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """
        Fetch company profiles (metadata) for the provided tickers.
        """
        return self._fetch(tickers=tickers, endpoint="profile")

    def fetch_ratios(self, tickers: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """
        Fetch TTM ratios for the provided tickers.
        """
        return self._fetch(tickers=tickers, endpoint="ratios-ttm")

    def fetch_key_metrics(self, tickers: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """
        Fetch TTM key metrics for the provided tickers.
        """
        return self._fetch(tickers=tickers, endpoint="key-metrics-ttm")

    # -------- internal helpers --------
    def _fetch(
        self,
        *,
        tickers: Sequence[str],
        endpoint: str,
    ) -> Dict[str, Dict[str, object]]:
        if not tickers:
            return {}
        results: Dict[str, Dict[str, object]] = {}
        canonical = [self._normalize_symbol(tk) for tk in tickers if tk]
        for chunk in _chunked(canonical, self.batch_size):
            for tk in chunk:
                fmp_symbol = self._to_fmp_symbol(tk)
                if self._should_skip(endpoint, fmp_symbol):
                    message = (
                        "Skipping FMP %s for %s due to recent %s response"
                        % (endpoint, tk, self._describe_last_error(endpoint, fmp_symbol))
                    )
                    if self._log_each_symbol:
                        log.info(message)
                    else:
                        log.debug(message)
                    continue
                payload = self._request(endpoint, params={"symbol": fmp_symbol})
                records = self._normalize_payload(payload)
                if not records:
                    if self._last_status_code == 402:
                        log.warning(
                            "FMP endpoint '%s' unavailable for %s under current plan; using empty payload.",
                            endpoint,
                            tk,
                        )
                        self._register_error(endpoint, fmp_symbol, self._last_status_code, "premium")
                        continue
                    if self._last_status_code and self._last_status_code >= 400:
                        log.warning(
                            "FMP endpoint '%s' returned status %s for %s; skipping.",
                            endpoint,
                            self._last_status_code,
                            tk,
                        )
                        self._register_error(endpoint, fmp_symbol, self._last_status_code, "error")
                        continue
                for item in records:
                    symbol = str(item.get("symbol") or "").upper()
                    ticker = tk
                    if symbol and symbol != fmp_symbol:
                        ticker = self._from_fmp_symbol(symbol)
                    if ticker:
                        results[ticker] = item
                        if self._log_each_symbol:
                            log.info("FMP %s fetched for %s", endpoint, ticker)
        return results

    def _request(self, endpoint: str, params: Optional[Dict[str, object]] = None) -> object:
        params = dict(params or {})
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._last_status_code = None
                self._sleep_backoff(attempt)
                continue
            self._last_call = time.monotonic()
            self._last_status_code = response.status_code
            if response.status_code == 429:
                wait = max(self.call_delay, 1.0)
                log.warning("FMP rate limit hit for %s; sleeping %.1fs", endpoint, wait)
                self._success_streak = 0
                if self._adapt_rate and self.call_delay < self._max_call_delay:
                    new_delay = min(self.call_delay * 1.5, self._max_call_delay)
                    if new_delay > self.call_delay:
                        log.info(
                            "Increasing FMP call delay from %.1fs to %.1fs after rate limit.",
                            self.call_delay,
                            new_delay,
                        )
                        self.call_delay = new_delay
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                log.warning("FMP server error %s for %s", response.status_code, endpoint)
                self._sleep_backoff(attempt)
                continue
            if response.status_code >= 400:
                log_fn = log.warning if response.status_code in {400, 401, 402, 403, 404} else log.error
                log_fn("FMP request failed (%s) for %s: %s", response.status_code, endpoint, response.text)
                self._success_streak = 0
                return []
            try:
                data = response.json()
            except ValueError as exc:
                last_error = exc
                log.warning("Failed to decode FMP response for %s: %s", endpoint, exc)
                self._sleep_backoff(attempt)
                self._success_streak = 0
                continue
            self._record_success()
            return data
        if last_error:
            self._last_status_code = None
            log.error("FMP request giving up for %s: %s", endpoint, last_error)
        return []

    def _should_skip(self, endpoint: str, symbol: str) -> bool:
        if self.error_retry_seconds <= 0:
            return False
        blocks = self._error_blocks.get(endpoint)
        if not blocks:
            return False
        expiry = blocks.get(symbol)
        if expiry is None:
            return False
        now = time.monotonic()
        if now < expiry:
            return True
        blocks.pop(symbol, None)
        if not blocks:
            self._error_blocks.pop(endpoint, None)
        return False

    def _register_error(self, endpoint: str, symbol: str, status: Optional[int], reason: str) -> None:
        if self.error_retry_seconds <= 0 or not status or status < 400:
            return
        block_until = time.monotonic() + self.error_retry_seconds
        self._error_blocks.setdefault(endpoint, {})[symbol] = block_until
        key = (endpoint, symbol)
        self._last_error_reason[key] = (status, reason)

    def _describe_last_error(self, endpoint: str, symbol: str) -> str:
        info = self._last_error_reason.get((endpoint, symbol))
        if not info:
            return "error"
        status, reason = info
        if reason == "premium":
            return f"premium ({status})"
        return f"status {status}"

    def _record_success(self) -> None:
        self._success_streak += 1
        if not self._adapt_rate or self.call_delay <= self._min_call_delay:
            return
        if self._success_streak >= self._success_threshold:
            new_delay = max(self._min_call_delay, self.call_delay * 0.9)
            if new_delay < self.call_delay:
                log.info(
                    "Reducing FMP call delay from %.1fs to %.1fs after %d clean calls.",
                    self.call_delay,
                    new_delay,
                    self._success_streak,
                )
                self.call_delay = new_delay
            self._success_streak = 0

    def _respect_rate_limit(self) -> None:
        if self.call_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        wait = self.call_delay - elapsed
        if wait > 0:
            time.sleep(wait)

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= self.max_retries:
            return
        delay = max(self.call_delay, 1.0) * attempt
        time.sleep(delay)

    @staticmethod
    def _normalize_payload(data: object) -> List[Dict[str, object]]:
        if isinstance(data, dict):
            if "Error Message" in data:
                log.warning("FMP error: %s", data.get("Error Message"))
                return []
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _normalize_symbol(ticker: str) -> str:
        return str(ticker or "").strip().upper()

    @staticmethod
    def _to_fmp_symbol(ticker: str) -> str:
        return ticker.replace("-", ".")

    @staticmethod
    def _from_fmp_symbol(symbol: str) -> str:
        return str(symbol or "").replace(".", "-").upper()
