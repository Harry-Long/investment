# mod/selector.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

def select_assets(prices: pd.DataFrame, policy: Dict) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Pre-optimization selection based on policy.portfolio.selection.
    Supports: top_by_1y_return, top_by_sharpe, none.
    Enforces min_count, max_count, target_count if provided.
    """
    port = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
    sel = (port.get("selection") or {}) if isinstance(port, dict) else {}
    method = str(sel.get("method", "none")).lower().strip()
    lookback = int(sel.get("lookback_days", 252))
    min_count = int(port.get("min_count", 1))
    max_count = int(port.get("max_count", len(prices.columns)))
    target_count = int(port.get("target_count", min(max_count, len(prices.columns))))
    target_count = max(min_count, min(target_count, max_count, len(prices.columns)))

    if method in ("none", "", "null"):
        return prices, list(prices.columns), "selection=none"

    rets = prices.pct_change().dropna()
    lb_rets = rets if rets.shape[0] < lookback else rets.iloc[-lookback:]
    ann_mu = lb_rets.mean() * 252.0
    ann_vol = lb_rets.std(ddof=1) * np.sqrt(252.0)
    sharpe = ann_mu / ann_vol.replace(0, np.nan)

    if method == "top_by_1y_return":
        rank = ann_mu.sort_values(ascending=False)
    elif method == "top_by_sharpe":
        rank = sharpe.sort_values(ascending=False)
    else:
        return prices, list(prices.columns), "selection=unknown -> skipped"

    chosen = list(rank.dropna().index[:target_count])
    if len(chosen) < min_count:
        remaining = [c for c in prices.columns if c not in chosen]
        chosen += remaining[: (min_count - len(chosen))]
    chosen = chosen[:max_count]

    note = f"selection={method}, lookback_days={lookback}, chosen={len(chosen)}/{len(prices.columns)}"
    return prices[chosen], chosen, note
