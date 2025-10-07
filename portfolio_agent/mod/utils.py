"""Utility helpers shared across portfolio_agent modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_zscore(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Clip outliers to quantile bounds then standardise to z-scores."""
    if series is None or series.empty:
        idx = getattr(series, "index", []) if series is not None else []
        return pd.Series(0.0, index=idx, dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    clipped = s.clip(lower=q_low, upper=q_high)
    mean = clipped.mean()
    std = clipped.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=clipped.index)
    return (clipped - mean) / std
