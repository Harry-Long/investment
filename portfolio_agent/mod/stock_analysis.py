"""
Candidate pool builder for US equities.

This module focuses on preparing a ~200 stock universe that feeds into the
existing selector/optimizer stack. It combines metadata filters, factor
scoring, sector/cap neutralisation and persistence helpers so that the
resulting list can be versioned on disk and reused as `universe.txt`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

try:  # Prefer project-level logging helper if available
    from .logging import get_logger  # type: ignore
except ImportError:  # pragma: no cover - fallback for standalone execution
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_logger(name: str) -> "logging.Logger":  # type: ignore
        return logging.getLogger(name)

try:
    from .utils import winsorize_zscore  # type: ignore
except ImportError:  # pragma: no cover
    from utils import winsorize_zscore  # type: ignore

try:  # pragma: no cover - optional class style provider
    from .data_provider import DataProvider as DefaultDataProvider  # type: ignore
except ImportError:
    DefaultDataProvider = None  # type: ignore

log = get_logger(__name__)


class DataProviderProtocol(Protocol):
    """
    Minimal interface expected from the data provider used by the pool builder.

    The concrete provider should expose point-in-time equity metadata together
    with historical prices, volumes and fundamental ratios.
    """

    def list_us_equities(self, as_of: dt.date) -> pd.DataFrame:
        """
        Return metadata for active US-listed equities as of `as_of`.

        Expected columns include (at minimum):
          ticker, name, sector, is_etf, is_adr, list_date, mcap, price, adv20
        """

    def get_history(
        self,
        tickers: Sequence[str],
        end: dt.date,
        lookback_days: int,
        field: str = "close",
    ) -> pd.DataFrame:
        """Return a price/volume panel indexed by date, columns=tickers."""

    def get_fundamentals(self, tickers: Sequence[str], as_of: dt.date) -> pd.DataFrame:
        """
        Return a DataFrame indexed by ticker with fundamental metrics such as:
        pe, pb, ev_to_ebitda, roe, gross_margin, profit_volatility,
        debt_to_equity.
        """

    def get_beta(
        self,
        tickers: Sequence[str],
        as_of: dt.date,
        lookback_days: int = 504,
        benchmark: Optional[str] = None,
    ) -> pd.Series:
        """Return trailing beta estimates versus the chosen benchmark."""


def _ensure_provider(provider: object, method: str) -> None:
    if provider is None or not hasattr(provider, method):
        raise AttributeError(
            f"Data provider is missing required method '{method}'. "
            "Please supply a provider implementing DataProviderProtocol."
        )


@dataclass
class BuilderSettings:
    cfg: Dict[str, object]
    target_size: int = field(init=False)
    tolerance: int = field(init=False)
    factor_weights: Dict[str, float] = field(init=False)
    neutralize_sector: bool = field(init=False)
    neutralize_cap: bool = field(init=False)
    max_per_sector: float = field(init=False)
    persistence: Dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        cand_cfg = self.cfg.get("candidate_pool")
        if not isinstance(cand_cfg, dict):
            raise ValueError("policy configuration missing 'candidate_pool' block.")
        self.target_size = int(cand_cfg.get("target_size", 200))
        self.tolerance = int(cand_cfg.get("tolerance", 10))
        self.factor_weights = {
            "momentum": float(cand_cfg.get("factors", {}).get("momentum", 0.3)),
            "quality": float(cand_cfg.get("factors", {}).get("quality", 0.25)),
            "value": float(cand_cfg.get("factors", {}).get("value", 0.2)),
            "stability": float(cand_cfg.get("factors", {}).get("stability", 0.15)),
            "liquidity": float(cand_cfg.get("factors", {}).get("liquidity", 0.1)),
        }
        neutralize = cand_cfg.get("neutralize", {}) if isinstance(cand_cfg.get("neutralize"), dict) else {}
        self.neutralize_sector = bool(neutralize.get("by_sector", True))
        self.neutralize_cap = bool(neutralize.get("cap_bucket", True))
        constraints = cand_cfg.get("constraints", {}) if isinstance(cand_cfg.get("constraints"), dict) else {}
        self.max_per_sector = float(constraints.get("max_per_sector", 0.25))
        persistence = cand_cfg.get("persistence", {}) if isinstance(cand_cfg.get("persistence"), dict) else {}
        default_dir = os.path.join("data", "universe")
        self.persistence = {
            "out_dir": persistence.get("out_dir", default_dir),
            "save_snapshot": bool(persistence.get("save_snapshot", True)),
            "write_symlink": bool(persistence.get("write_symlink", True)),
        }


class CandidatePoolBuilder:
    """Core engine producing the candidate pool DataFrame."""

    def __init__(self, policy_cfg: Dict[str, object], provider: DataProviderProtocol):
        self.policy_cfg = policy_cfg
        self.settings = BuilderSettings(policy_cfg)
        self.provider = provider

    # -------- public API --------
    def build(self, as_of: Optional[dt.date] = None) -> pd.DataFrame:
        """Run the full pipeline: filters → factors → scoring → persistence."""
        as_of = as_of or dt.date.today()
        log.info("Building candidate pool as of %s", as_of.isoformat())
        base = self._load_base_universe(as_of)
        filt = self._apply_base_filters(base, as_of)
        fac = self._compute_factors(filt, as_of)
        scored = self._score_and_select(fac)
        self._persist(scored, as_of, manual=False)
        return scored

    def update_manual(
        self,
        add: Optional[Iterable[str]] = None,
        remove: Optional[Iterable[str]] = None,
        as_of: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        """
        Manually adjust the latest universe by adding/removing tickers.
        Snapshots the result using today's date (unless overridden).
        """
        as_of = as_of or dt.date.today()
        current = self._read_current_universe()
        updated = set(current)
        if add:
            updated.update([t.strip() for t in add if t])
        if remove:
            to_remove = {t.strip() for t in remove if t}
            updated = {t for t in updated if t not in to_remove}
        ordered = sorted(updated)
        df = pd.DataFrame({"ticker": ordered})
        self._persist(df, as_of, manual=True)
        log.info("Manual candidate pool update complete (%d tickers).", len(df))
        return df

    # -------- internal helpers --------
    def _load_base_universe(self, as_of: dt.date) -> pd.DataFrame:
        _ensure_provider(self.provider, "list_us_equities")
        meta = self.provider.list_us_equities(as_of=as_of)
        if meta is None or meta.empty:
            raise RuntimeError("Data provider returned empty metadata universe.")
        required_cols = {"ticker", "sector", "is_etf", "is_adr", "list_date", "mcap", "price"}
        missing = required_cols.difference(meta.columns)
        if missing:
            raise ValueError(f"list_us_equities missing required columns: {sorted(missing)}")
        meta = meta.copy()
        meta["ticker"] = meta["ticker"].astype(str).str.upper()
        meta = meta.drop_duplicates(subset=["ticker"]).set_index("ticker")
        return meta

    def _apply_base_filters(self, df: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
        cand_cfg = self.policy_cfg["candidate_pool"]
        filters = cand_cfg.get("base_filters", {}) if isinstance(cand_cfg.get("base_filters"), dict) else {}
        exclude = cand_cfg.get("exclude", {}) if isinstance(cand_cfg.get("exclude"), dict) else {}
        out = df.copy()

        if exclude.get("etf", True) and "is_etf" in out.columns:
            out = out.loc[~out["is_etf"]]
        if exclude.get("adr", True) and "is_adr" in out.columns:
            out = out.loc[~out["is_adr"]]
        if exclude.get("penny", True):
            if "price" in out.columns and not out["price"].dropna().empty:
                threshold = max(float(filters.get("min_price", 5.0)), 1.0)
                out = out.loc[out["price"] >= threshold]
            else:
                log.warning("Price data unavailable; skipping penny stock filter.")
        elif "min_price" in filters:
            if "price" in out.columns and not out["price"].dropna().empty:
                out = out.loc[out["price"] >= float(filters["min_price"])]
            else:
                log.warning("min_price filter requested but price data missing; skipping.")

        if "min_mcap" in filters:
            if "mcap" not in out.columns or out["mcap"].dropna().empty:
                log.warning("min_mcap filter requested but 'mcap' data missing; skipping.")
            else:
                out = out.loc[out["mcap"] >= float(filters["min_mcap"])]

        if "min_adv20" in filters:
            if "adv20" not in out.columns or out["adv20"].dropna().empty:
                log.warning("min_adv20 filter requested but 'adv20' data missing; skipping.")
            else:
                out = out.loc[out["adv20"] >= float(filters["min_adv20"])]

        if filters.get("min_days_listed"):
            if "list_date" in out.columns:
                list_dates = pd.to_datetime(out["list_date"], errors="coerce")
                if list_dates.notna().any():
                    days_listed = (pd.Timestamp(as_of) - list_dates).dt.days
                    out = out.loc[days_listed >= int(filters["min_days_listed"])]
                else:
                    log.warning("list_date data unavailable; skipping min_days_listed filter.")
            else:
                log.warning("min_days_listed filter requested but 'list_date' column missing; skipping.")

        if out.empty:
            raise RuntimeError("Base filters removed all candidates. Revisit thresholds.")

        columns_to_keep = [
            c
            for c in out.columns
            if c
            not in {
                "is_spac",
                "is_otc",
            }
        ]
        filtered = out[columns_to_keep].copy()
        log.info("Base filters retained %d tickers.", len(filtered))
        return filtered

    def _compute_factors(self, df: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
        tickers = df.index.tolist()
        if not tickers:
            raise RuntimeError("No tickers available post filtering.")

        _ensure_provider(self.provider, "get_history")
        prices = self.provider.get_history(tickers, end=as_of, lookback_days=400, field="close")
        volumes = self.provider.get_history(tickers, end=as_of, lookback_days=400, field="volume")

        if prices is None or prices.empty:
            raise RuntimeError("Price history request returned empty frame.")

        prices = prices.sort_index().ffill()
        volumes = volumes.sort_index().ffill() if volumes is not None else None

        returns = prices.pct_change()
        mom_6_1 = prices.pct_change(126).shift(21).iloc[-1]
        mom_6_1.name = "momentum"

        # Stability metrics
        vol_63 = returns.rolling(63).std().iloc[-1]
        vol_63.name = "volatility"

        _ensure_provider(self.provider, "get_beta")
        beta = self.provider.get_beta(tickers, as_of=as_of, lookback_days=504)
        if beta is None or isinstance(beta, float):
            beta = pd.Series(beta, index=tickers, name="beta")
        else:
            beta = beta.reindex(tickers).rename("beta")

        # Liquidity from rolling dollar volume
        if volumes is not None and not volumes.empty:
            dollar_vol = (prices * volumes).rolling(20).mean().iloc[-1]
        else:
            dollar_vol = df.get("adv20", pd.Series(index=tickers, dtype=float))
        dollar_vol = pd.Series(dollar_vol, index=tickers, name="liquidity")

        _ensure_provider(self.provider, "get_fundamentals")
        fundamentals = self.provider.get_fundamentals(tickers, as_of=as_of)
        if fundamentals is None or fundamentals.empty:
            fundamentals = pd.DataFrame(index=tickers)
        fundamentals = fundamentals.reindex(tickers)

        tickers_index = pd.Index(tickers, name="ticker")
        inv_ev = _safe_inverse(_get_series(fundamentals, "ev_to_ebitda", tickers_index))
        if inv_ev.isna().all():
            inv_ev = _safe_inverse(_get_series(fundamentals, "ev_ebitda", tickers_index))
        value_components = pd.DataFrame(
            {
                "inv_pe": _safe_inverse(_get_series(fundamentals, "pe", tickers_index)),
                "inv_pb": _safe_inverse(_get_series(fundamentals, "pb", tickers_index)),
                "inv_ev_ebitda": inv_ev,
                "fcf_yield": _get_series(fundamentals, "fcf_yield", tickers_index),
            }
        )
        value_score = value_components.mean(axis=1).rename("value").fillna(0.0)

        quality_components = pd.DataFrame(
            {
                "roe": _get_series(fundamentals, "roe", tickers_index),
                "roic": _get_series(fundamentals, "roic", tickers_index),
                "gross_margin": _get_series(fundamentals, "gross_margin", tickers_index),
                "profit_stability": -_get_series(fundamentals, "profit_volatility", tickers_index),
                "debt_load": -_get_series(fundamentals, "debt_to_equity", tickers_index),
                "net_debt_to_ebitda": -_get_series(fundamentals, "net_debt_to_ebitda", tickers_index),
            }
        )
        quality_score = quality_components.mean(axis=1).rename("quality").fillna(0.0)

        stability_components = pd.DataFrame(
            {
                "low_vol": -vol_63,
                "low_beta": -beta,
            }
        )
        stability_score = stability_components.mean(axis=1).rename("stability")

        liquidity_score = dollar_vol.rename("liquidity")

        factors = pd.concat(
            [
                mom_6_1,
                quality_score,
                value_score,
                stability_score,
                liquidity_score,
            ],
            axis=1,
        )

        factors = factors.apply(lambda s: winsorize_zscore(s, lower=0.01, upper=0.99), axis=0)
        factors.replace([np.inf, -np.inf], np.nan, inplace=True)
        factors = factors.fillna(0.0)

        enriched = df.join(factors, how="inner")
        if enriched.empty:
            raise RuntimeError("Factor computation left no usable tickers.")

        if self.settings.neutralize_sector and "sector" in enriched.columns:
            enriched = self._neutralize_by_group(enriched, group_col="sector")

        if self.settings.neutralize_cap:
            enriched = self._neutralize_cap_bucket(enriched)

        enriched = enriched.dropna(subset=list(self.settings.factor_weights))
        log.info("Factor computation finished for %d tickers.", len(enriched))
        return enriched

    def _neutralize_by_group(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        numeric_cols = list(self.settings.factor_weights.keys())
        grouped = df.groupby(group_col, dropna=False)
        adjusted_frames: List[pd.DataFrame] = []
        for _, sub in grouped:
            to_scale = sub[numeric_cols]
            centered = to_scale - to_scale.mean()
            std = to_scale.std(ddof=0).replace(0, np.nan)
            scaled = centered.divide(std, axis=1).fillna(0.0)
            adjusted = sub.copy()
            adjusted[numeric_cols] = scaled
            adjusted_frames.append(adjusted)
        return pd.concat(adjusted_frames).reindex(df.index)

    def _neutralize_cap_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        if "mcap" not in df.columns:
            return df
        quantiles = df["mcap"].rank(method="first", pct=True)
        buckets = pd.cut(quantiles, bins=[0, 0.33, 0.66, 1.0], labels=["small", "mid", "large"])
        df = df.copy()
        df["cap_bucket"] = buckets
        df = self._neutralize_by_group(df, group_col="cap_bucket")
        df.drop(columns=["cap_bucket"], inplace=True)
        return df

    def _score_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.settings.factor_weights
        df = df.copy()
        df["score"] = sum(df[factor] * weight for factor, weight in weights.items())
        df.sort_values("score", ascending=False, inplace=True)

        target = self.settings.target_size
        tolerance = self.settings.tolerance
        max_per_sector = int(np.ceil(target * self.settings.max_per_sector))

        if "sector" in df.columns and max_per_sector > 0:
            df = (
                df.groupby("sector", group_keys=False)
                .head(max_per_sector)
                .sort_values("score", ascending=False)
            )

        top_n = df.head(target + tolerance).copy()
        final = top_n.iloc[: target + tolerance]

        if len(final) < target - tolerance:
            log.warning(
                "Candidate pool size (%d) below target-tolerance (%d). Review filters.",
                len(final),
                target - tolerance,
            )
        log.info("Candidate pool selected %d tickers.", len(final))
        return final.reset_index().rename(columns={"index": "ticker"})

    def _persist(self, df: pd.DataFrame, as_of: dt.date, manual: bool) -> None:
        settings = self.settings.persistence
        out_dir = settings["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        tag = "manual" if manual else "auto"
        snapshot_name = f"universe_{as_of.strftime('%Y%m%d')}_{tag}.txt"
        snapshot_path = os.path.join(out_dir, snapshot_name)

        if settings.get("save_snapshot", True):
            df["ticker"].to_csv(snapshot_path, index=False, header=False)
            log.info("Universe snapshot written to %s", snapshot_path)

        current_path = os.path.join(out_dir, "universe.txt")
        if settings.get("write_symlink", True):
            try:
                if os.path.islink(current_path) or os.path.exists(current_path):
                    os.unlink(current_path)
                os.symlink(snapshot_name, current_path)
                log.info("Universe symlink updated -> %s", snapshot_name)
                return
            except OSError:
                log.debug("Symlink failed, falling back to copy for %s", current_path)
        df["ticker"].to_csv(current_path, index=False, header=False)
        log.info("Universe file replaced at %s", current_path)

    def _read_current_universe(self) -> List[str]:
        out_dir = self.settings.persistence["out_dir"]
        current = os.path.join(out_dir, "universe.txt")
        if not os.path.exists(current):
            return []
        series = pd.read_csv(current, header=None)[0]
        return [str(t).strip().upper() for t in series.tolist() if isinstance(t, str)]


# --------- helper utilities ---------

def _get_series(frame: Optional[pd.DataFrame], key: str, index: pd.Index) -> pd.Series:
    if frame is None or key not in frame.columns:
        return pd.Series(np.nan, index=index, dtype=float)
    series = pd.to_numeric(frame[key], errors="coerce")
    return series.reindex(index)

def _safe_inverse(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    inv = 1.0 / numeric
    return inv


def build_candidate_pool(
    policy_cfg: Dict[str, object],
    provider: DataProviderProtocol,
    as_of: Optional[dt.date] = None,
) -> pd.DataFrame:
    """Convenience wrapper for one-off builds."""
    builder = CandidatePoolBuilder(policy_cfg=policy_cfg, provider=provider)
    return builder.build(as_of=as_of)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate pool builder")
    parser.add_argument("--policy", default="policy.yaml", help="Path to policy configuration.")
    parser.add_argument("--task", choices=["build", "update"], default="build")
    parser.add_argument("--as-of", help="Snapshot date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--add", nargs="*", help="Tickers to add during manual update.")
    parser.add_argument("--remove", nargs="*", help="Tickers to remove during manual update.")
    return parser.parse_args()


def _load_policy(path: str) -> Dict[str, object]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = _parse_cli_args()
    policy_cfg = _load_policy(args.policy)
    if DefaultDataProvider is None:
        raise RuntimeError(
            "Default DataProvider class not available. Supply a provider "
            "implementation or integrate the builder within your application."
        )
    provider = DefaultDataProvider(policy_cfg)  # type: ignore[misc]
    as_of = dt.datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    builder = CandidatePoolBuilder(policy_cfg, provider)
    if args.task == "build":
        builder.build(as_of=as_of)
    else:
        builder.update_manual(add=args.add, remove=args.remove, as_of=as_of)


if __name__ == "__main__":  # pragma: no cover
    main()
