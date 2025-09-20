# mod/reporting_extras.py
import os, pandas as pd, matplotlib.pyplot as plt

def save_core_csvs(pf_ret, nav, rc, corr, outdir):
    os.makedirs(outdir, exist_ok=True)
    pf_ret.to_frame("portfolio_return").to_csv(os.path.join(outdir,"portfolio_returns.csv"))
    nav.to_frame("nav").to_csv(os.path.join(outdir,"portfolio_nav.csv"))
    rc.to_frame("risk_contribution").to_csv(os.path.join(outdir,"risk_contribution.csv"))
    corr.to_csv(os.path.join(outdir,"correlation_matrix.csv"))

def plot_nav(nav, outdir):
    plt.figure(figsize=(8,4)); plt.plot(nav.index, nav.values); plt.title("Portfolio NAV"); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"nav.png"),dpi=150); plt.close()

def plot_corr(corr, outdir):
    plt.figure(figsize=(5,4)); im=plt.imshow(corr.values, aspect="auto"); plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"corr.png"),dpi=150); plt.close()

def plot_prices(prices, outdir, suffix="", benchmark=None, benchmark_label="Benchmark"):
    plt.figure(figsize=(9,4.5))
    for col in prices.columns: plt.plot(prices.index, prices[col].values, label=str(col))
    if benchmark is not None and getattr(benchmark,'size',0)>0:
        plt.plot(benchmark.index, benchmark.values, linestyle='--', linewidth=1.2, label=str(benchmark_label))
    plt.title("Price Levels"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"prices{suffix}.png"),dpi=150); plt.close()

def make_text_report(tickers, weights, ann_return, ann_vol, sharpe, mdd_tuple, var95, var99, relative_metrics=None):
    peak,trough,mdd=mdd_tuple; lines=[]; lines.append("Portfolio Analysis Summary (Educational Only)"); lines.append("-------------------------------------------")
    lines.append(f"Tickers: {list(tickers)}"); lines.append(f"Weights: {dict(weights)}"); lines.append("")
    lines.append("Performance and Risk")
    lines.append(f"Annualized Return: {ann_return:.4f}"); lines.append(f"Annualized Volatility: {ann_vol:.4f}")
    lines.append(f"Sharpe: {sharpe:.4f}"); lines.append(f"Max Drawdown: {mdd:.4f}  Window: {peak} to {trough}")
    lines.append(""); lines.append(f"95% VaR: {var95['VaR']:.4f}   95% ES: {var95['ES']:.4f}")
    lines.append(f"99% VaR: {var99['VaR']:.4f}   99% ES: {var99['ES']:.4f}")
    if relative_metrics:
        lines.append(""); lines.append("Relative to benchmark")
        if 'ann_excess' in relative_metrics: lines.append(f"Annualized Excess Return: {relative_metrics['ann_excess']:.4f}")
        if 'tracking_error' in relative_metrics: lines.append(f"Tracking Error (ann.): {relative_metrics['tracking_error']:.4f}")
        if 'information_ratio' in relative_metrics: lines.append(f"Information Ratio: {relative_metrics['information_ratio']:.4f}")
    lines.append(""); lines.append("Disclaimer: For education only. Not investment advice."); return "\n".join(lines)

def max_drawdown_from_nav(nav):
    peak=nav.cummax(); dd=nav/peak-1.0; i_trough=dd.idxmin(); i_peak=peak.loc[:i_trough].idxmax(); return str(i_peak), str(i_trough), float(dd.min())

def compute_relative_metrics(pf_ret, bench_ret, periods_per_year=252):
    idx=pf_ret.index.intersection(bench_ret.index)
    if len(idx)==0: return {}
    ex=(pf_ret.loc[idx]-bench_ret.loc[idx]).dropna()
    if ex.empty: return {}
    ann_excess=(1+ex).prod()**(periods_per_year/len(ex))-1.0
    te=ex.std(ddof=1)*(periods_per_year**0.5)
    ir=(ann_excess/te) if te>0 else float('nan')
    return {"ann_excess":float(ann_excess),"tracking_error":float(te),"information_ratio":float(ir)}

# --- NEW: combined report for two strategies ---
import base64, io

def save_dual_report(
    ret_a: pd.Series,
    ret_b: pd.Series,
    label_a: str = "Equal Weights",
    label_b: str = "Optimized",
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    title: str = "Combined Strategy Report",
    out_html: str = "output/qs_report_combined.html",
):
    """
    Create a single HTML report that overlays two strategy curves and prints both metric tables.
    - ret_a / ret_b: daily simple returns (aligned or partially overlapping is fine)
    - benchmark: optional daily returns for overlay/reference
    """
    # Align indices safely
    df = pd.DataFrame({label_a: ret_a, label_b: ret_b}).dropna(how="all")
    if benchmark is not None and getattr(benchmark, "size", 0) > 0:
        df[benchmark_label] = benchmark.reindex(df.index)

    # Build NAVs
    nav = (1 + df[[label_a, label_b]].fillna(0)).cumprod()
    if benchmark is not None and benchmark_label in df.columns:
        nav_bm = (1 + df[[benchmark_label]].fillna(0)).cumprod()
    else:
        nav_bm = None

    # Plot overlay
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nav.plot(ax=ax)
    if nav_bm is not None:
        nav_bm.plot(ax=ax, style="--", linewidth=1.2)
    ax.set_title("Cumulative Performance")
    ax.set_ylabel("Growth of 1")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    # Save plot to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Compute metrics using QuantStats basic mode
    from .qs_wrapper import basic_metrics
    # Metrics: compute individually to avoid NaNs due to alignment quirks
    m_a = basic_metrics(df[label_a].dropna(), benchmark=df[benchmark_label].dropna() if benchmark is not None and benchmark_label in df else None)
    m_b = basic_metrics(df[label_b].dropna(), benchmark=df[benchmark_label].dropna() if benchmark is not None and benchmark_label in df else None)

    # Render to HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 20px; }}
    .row {{ display: flex; gap: 24px; }}
    .col {{ flex: 1; min-width: 360px; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 24px; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th {{ background: #f6f6f6; }}
    .note {{ color: #666; font-size: 12px; margin-top: 8px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">This page overlays two strategies and shows both metric tables. Educational use only.</div>
  <h2>Performance Overlay</h2>
  <img src="data:image/png;base64,{img_b64}" alt="overlay plot" />

  <div class="row">
    <div class="col">
      <h2>Metrics — {label_a}</h2>
      {m_a.to_html(border=0, classes="table", justify="right")}
    </div>
    <div class="col">
      <h2>Metrics — {label_b}</h2>
      {m_b.to_html(border=0, classes="table", justify="right")}
    </div>
  </div>

  <div class="note">
    Benchmark: {"None" if benchmark is None else benchmark_label}. Tables are QuantStats basic metrics.
  </div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
