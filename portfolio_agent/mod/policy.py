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

def resolve_dates_and_benchmark(args, policy: Dict):
    """Resolve start, end, benchmark from policy.data with CLI fallback."""
    pol_data = policy.get("data") if isinstance(policy, dict) else {}
    start = args.start if args.source == "stooq" else args.synthetic_start
    end = args.end
    if args.source == "stooq":
        if pol_data.get("start"):
            start = str(pol_data.get("start"))
        if pol_data.get("end") not in (None, "", "null"):
            end = str(pol_data.get("end"))
    bench = args.benchmark
    if isinstance(pol_data.get("benchmark"), str) and pol_data.get("benchmark"):
        bench = pol_data.get("benchmark")
    return start, end, bench
