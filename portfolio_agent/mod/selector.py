# mod/selector.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional


def load_candidate_universe(path: Optional[str] = None) -> List[str]:
    """
    Load tickers from the candidate universe text file.

    Parameters
    ----------
    path : str, optional
        Explicit path to the universe file. Defaults to data/universe/universe.txt.
    """
    target = path or os.path.join("data", "universe", "universe.txt")
    if not os.path.exists(target):
        return []
    tickers = pd.read_csv(target, header=None)[0].astype(str).str.strip().tolist()
    return [t for t in tickers if t]

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

    note_parts: List[str] = []

    # Optional candidate universe gating
    cand_cfg = policy.get("candidate_pool") if isinstance(policy, dict) else {}
    if isinstance(cand_cfg, dict):
        persistence = cand_cfg.get("persistence", {}) if isinstance(cand_cfg.get("persistence"), dict) else {}
        universe_path = persistence.get("current_path")
        if not universe_path:
            out_dir = persistence.get("out_dir") or os.path.join("data", "universe")
            universe_path = os.path.join(out_dir, "universe.txt")
        candidate = load_candidate_universe(universe_path)
        if candidate:
            cols = [c for c in prices.columns if c in candidate]
            if cols:
                removed = len(prices.columns) - len(cols)
                if removed > 0:
                    note_parts.append(f"candidate_gate=filtered_{removed}")
                prices = prices[cols]
            else:
                note_parts.append("candidate_gate=empty_intersection")

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

    base_note = f"selection={method}, lookback_days={lookback}, chosen={len(chosen)}/{len(prices.columns)}"
    if note_parts:
        base_note = ", ".join(note_parts + [base_note])
    note = base_note
    return prices[chosen], chosen, note
