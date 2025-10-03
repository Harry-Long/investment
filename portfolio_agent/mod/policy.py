# mod/policy.py
from __future__ import annotations
import os
from typing import Dict, Tuple, Optional

def load_policy(path: Optional[str]) -> Dict:
    """Load YAML policy into a dict. Returns {} if not found or parse error."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[warn] Failed to read policy file {path}: {e}")
        return {}

def resolve_mode(cli_mode: Optional[str], policy: Dict, src_label: str) -> Tuple[str, str]:
    """Resolve portfolio mode from CLI or policy. Returns (mode_in_use, note)."""
    pol_port = policy.get("portfolio") if isinstance(policy.get("portfolio"), dict) else {}
    yaml_mode = pol_port.get("mode") if pol_port else policy.get("mode")
    yaml_mode = (yaml_mode or "naive")
    mode_in_use = cli_mode if cli_mode is not None else yaml_mode
    note = "overridden by CLI" if cli_mode is not None else f"from {src_label}"
    return mode_in_use, note

def resolve_dates_and_benchmark(policy: Dict, source: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve start, end, and benchmark tickers from the policy.

    Parameters
    ----------
    policy: Dict
        Parsed YAML configuration.
    source: str
        Price source identifier ("stooq" or "synthetic").
    """

    data = policy.get("data", {}) if isinstance(policy, dict) else {}

    if source == "stooq":
        start = data.get("start") or "2018-01-01"
        end_val = data.get("end")
        end = None if end_val in (None, "", "null") else str(end_val)
    else:
        synthetic_cfg = data.get("synthetic") if isinstance(data.get("synthetic"), dict) else {}
        start = data.get("synthetic_start") or synthetic_cfg.get("start") or "2023-01-03"
        end = None

    bench = data.get("benchmark") or policy.get("benchmark")
    bench = bench.strip() if isinstance(bench, str) else bench

    return (str(start) if start is not None else None, end, bench)
