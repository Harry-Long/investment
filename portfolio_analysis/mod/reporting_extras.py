import os
import pandas as pd
import matplotlib.pyplot as plt

def save_core_csvs(pf_ret: pd.Series, nav: pd.Series, rc: pd.Series, corr: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    pf_ret.to_frame("portfolio_return").to_csv(os.path.join(outdir, "portfolio_returns.csv"))
    nav.to_frame("nav").to_csv(os.path.join(outdir, "portfolio_nav.csv"))
    rc.to_frame("risk_contribution").to_csv(os.path.join(outdir, "risk_contribution.csv"))
    corr.to_csv(os.path.join(outdir, "correlation_matrix.csv"))

def plot_nav(nav: pd.Series, outdir: str):
    plt.figure(figsize=(8,4))
    plt.plot(nav.index, nav.values)
    plt.title("Portfolio NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nav.png"), dpi=150)
    plt.close()

def plot_corr(corr: pd.DataFrame, outdir: str):
    plt.figure(figsize=(5,4))
    im = plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "corr.png"), dpi=150)
    plt.close()

def make_text_report(tickers, weights, ann_return, ann_vol, sharpe, mdd_tuple, var95, var99) -> str:
    peak, trough, mdd = mdd_tuple
    lines = []
    lines.append("Portfolio Analysis Summary (Educational Only)")
    lines.append("-------------------------------------------")
    lines.append(f"Tickers: {list(tickers)}")
    lines.append(f"Weights: {dict(weights)}")
    lines.append("")
    lines.append("Performance and Risk")
    lines.append(f"Annualized Return: {ann_return:.4f}")
    lines.append(f"Annualized Volatility: {ann_vol:.4f}")
    lines.append(f"Sharpe: {sharpe:.4f}")
    lines.append(f"Max Drawdown: {mdd:.4f}  Window: {peak} to {trough}")
    lines.append("")
    lines.append(f"95% VaR: {var95['VaR']:.4f}   95% ES: {var95['ES']:.4f}")
    lines.append(f"99% VaR: {var99['VaR']:.4f}   99% ES: {var99['ES']:.4f}")
    lines.append("")
    lines.append("Disclaimer: For education only. Not investment advice.")
    return "\n".join(lines)

def max_drawdown_from_nav(nav: pd.Series):
    peak = nav.cummax()
    dd = nav / peak - 1.0
    i_trough = dd.idxmin()
    i_peak = peak.loc[:i_trough].idxmax()
    return str(i_peak), str(i_trough), float(dd.min())


def plot_prices(prices: pd.DataFrame, outdir: str, suffix: str = "", benchmark: pd.Series | None = None, benchmark_label: str = "Benchmark"):
    plt.figure(figsize=(9,4.5))
    for col in prices.columns:
        plt.plot(prices.index, prices[col].values, label=str(col))
    plt.title("Price Levels")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fn = os.path.join(outdir, f"prices{suffix}.png")
    plt.savefig(fn, dpi=150)
    plt.close()


def compute_relative_metrics(pf_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 252) -> dict:
    # align indices
    idx = pf_ret.index.intersection(bench_ret.index)
    if len(idx) == 0:
        return {}
    ex = (pf_ret.loc[idx] - bench_ret.loc[idx]).dropna()
    if ex.empty:
        return {}
    ann_excess = (1 + ex).prod() ** (periods_per_year / len(ex)) - 1.0
    te = ex.std(ddof=1) * (periods_per_year ** 0.5)  # tracking error (ann.)
    ir = (ann_excess / te) if te > 0 else float("nan")
    return {"ann_excess": float(ann_excess), "tracking_error": float(te), "information_ratio": float(ir)}
