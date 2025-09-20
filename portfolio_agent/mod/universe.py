# mod/universe.py
from __future__ import annotations
import os, csv
from typing import List, Tuple, Dict, Optional

def _load_universe_from_file(path: str) -> List[str]:
    """Load tickers from txt or csv. Supports comments with '#', comma or space separated lines."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"universe_file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    tickers: List[str] = []
    if ext in [".csv"]:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if rdr.fieldnames:
                cols_lower = [c.lower() for c in rdr.fieldnames]
                if "ticker" in cols_lower:
                    col = rdr.fieldnames[cols_lower.index("ticker")]
                else:
                    col = rdr.fieldnames[0]
                for row in rdr:
                    v = row.get(col)
                    if v: tickers.append(str(v).strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                tickers.extend(parts)
    seen=set(); uniq=[]
    for t in tickers:
        if t and t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

def resolve_universe(cli_tickers: Optional[list], policy: Dict) -> Tuple[list, str]:
    """
    Resolution order:
      1) CLI --tickers
      2) policy.data.universe_file
      3) policy.data.universe or top-level universe
    """
    if cli_tickers:
        return list(cli_tickers), "from CLI --tickers"
    data = policy.get("data", {}) if isinstance(policy, dict) else {}
    uf = data.get("universe_file") or policy.get("universe_file")
    if isinstance(uf, str) and uf.strip():
        return _load_universe_from_file(uf.strip()), f"from universe_file={uf}"
    universe = (data.get("universe") if isinstance(data, dict) else None) or policy.get("universe")
    if isinstance(universe, (list, tuple)) and len(universe) > 0:
        return list(universe), "from policy.yaml universe"
    raise RuntimeError("No tickers provided. Use --tickers, or set data.universe_file, or data.universe in policy.yaml")
