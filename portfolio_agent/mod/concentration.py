# mod/concentration.py
from __future__ import annotations
import pandas as pd

def enforce_topk_share(w: pd.Series, topk: int = 5, min_share: float = 0.50, upper: float = 0.35) -> pd.Series:
    """Clip to [0, upper], then scale top-k up and others down to meet min_share. Renormalize to 1."""
    w = w.clip(lower=0, upper=upper)
    s = w.sort_values(ascending=False)
    total = s.sum()
    if total <= 0:
        return w
    s = s / total
    c_top = s.iloc[:topk].sum()
    if c_top >= min_share:
        return s.reindex(w.index)
    need = min_share - c_top
    rest = 1 - c_top
    if rest <= 0 or need <= 0:
        return s.reindex(w.index)
    scale_top = (c_top + need) / c_top
    scale_oth = (rest - need) / rest
    s.iloc[:topk] = (s.iloc[:topk] * scale_top).clip(upper=upper)
    s.iloc[topk:] = (s.iloc[topk:] * scale_oth).clip(lower=0)
    s = s / s.sum()
    return s.reindex(w.index)
