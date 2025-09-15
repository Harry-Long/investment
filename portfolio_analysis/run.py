# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np

from mod.data_provider import get_prices_synthetic, get_prices_stooq
from mod.qs_wrapper import basic_metrics, save_html_report
from mod.pypfopt_wrapper import optimize_max_sharpe
from mod.risk_tools import to_simple_returns, portfolio_returns, annualized_cov, risk_contribution, corr_matrix, var_es_hist, portfolio_nav
from mod.reporting_extras import save_core_csvs, plot_nav, plot_corr, plot_prices, make_text_report, max_drawdown_from_nav, compute_relative_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Portfolio analysis with QuantStats + PyPortfolioOpt (Stooq backend only)")
    p.add_argument("--source", choices=["synthetic","stooq"], default="synthetic")
    p.add_argument("--tickers", nargs="+", default=["AAPL.US","MSFT.US"], help="Use .US suffix for US tickers")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="Initial weights; default is equal weight")
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date for Stooq")
    p.add_argument("--end", type=str, default=None, help="End date for Stooq (None = today)")
    p.add_argument("--synthetic-start", type=str, default="2023-01-03", help="Synthetic start date")
    p.add_argument("--synthetic-periods", type=int, default=252*2, help="Synthetic business-day periods")
    p.add_argument("--outdir", type=str, default="output")
    p.add_argument("--title", type=str, default="QuantStats Report")
    p.add_argument("--alpha", type=float, default=0.95, help="Confidence level for VaR/ES")
    p.add_argument("--benchmark", type=str, default="SPY.US", help="Benchmark ticker from Stooq, e.g., SPY.US; set empty to disable")
    p.add_argument("--export-extras", action="store_true", help="Export NAV, RC, Corr, VaR/ES, and a text summary")
    return p.parse_args()

def normalize_weights(tickers, weights):
    if weights is None:
        return pd.Series([1.0/len(tickers)]*len(tickers), index=tickers)
    if len(weights) != len(tickers):
        raise ValueError("Length of weights must match tickers")
    w = pd.Series(weights, index=tickers)
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be > 0")
    return w / s

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    suffix = "_synth" if args.source == "synthetic" else "_real"

    if args.source == "synthetic":
        prices = get_prices_synthetic(tuple(args.tickers), start=args.synthetic_start, periods=args.synthetic_periods)
    else:
        prices = get_prices_stooq(tuple(args.tickers), start=args.start, end=args.end)

    # Optional benchmark (Stooq)
    bench_ret = None
    if args.source == "stooq" and args.benchmark and len(args.benchmark.strip()) > 0:
        try:
            bdf = get_prices_stooq((args.benchmark.strip(),), start=args.start, end=args.end)
            if not bdf.empty:
                from mod.risk_tools import to_simple_returns
                bench_ret = to_simple_returns(bdf).iloc[:, 0]
        except Exception as e:
            print(f"[warn] Failed to fetch benchmark {args.benchmark}: {e}")

    ret = to_simple_returns(prices)
    if ret.empty or ret.shape[0] < 30:
        raise RuntimeError(f"Insufficient data rows: {ret.shape[0]}. Check tickers and date range.")

    w0 = normalize_weights(args.tickers, args.weights)
    pf_ret_0 = portfolio_returns(ret, w0)
    if pf_ret_0.empty:
        raise RuntimeError("Portfolio return series is empty.")

    # QuantStats outputs
    metrics_df = basic_metrics(pf_ret_0, benchmark=bench_ret)
    metrics_csv = os.path.join(args.outdir, f"qs_metrics{suffix}.csv")
    metrics_df.to_csv(metrics_csv, encoding="utf-8")
    html_path = os.path.join(args.outdir, f"qs_report{suffix}.html")
    save_html_report(pf_ret_0, html_path, title=args.title, benchmark=bench_ret)

    # PyPortfolioOpt optimization
    opt = optimize_max_sharpe(prices, risk_free_rate=0.0, l2_gamma=0.001)
    w_opt = pd.Series(opt["weights"]).reindex(prices.columns).fillna(0.0)

    # Save weights and returns
    if bench_ret is not None:
        bench_csv = os.path.join(args.outdir, f"benchmark_returns{suffix}.csv")
        bench_ret.to_frame("benchmark_return").to_csv(bench_csv)
    w0.to_frame("weight").to_csv(os.path.join(args.outdir, f"weights_initial{suffix}.csv"))
    w_opt.to_frame("weight").to_csv(os.path.join(args.outdir, f"weights_optimized{suffix}.csv"))
    pf_ret_0.to_frame("ret_initial").to_csv(os.path.join(args.outdir, f"returns_initial{suffix}.csv"))
    pf_ret_opt = portfolio_returns(ret, w_opt)
    pf_ret_opt.to_frame("ret_optimized").to_csv(os.path.join(args.outdir, f"returns_optimized{suffix}.csv"))

    print("Initial weights")
    print(w0.round(4).to_string())
    print("\nOptimized weights (PyPortfolioOpt)")
    print(w_opt.round(4).to_string())
    print("\nOptimized performance (estimate)")
    print({k: round(v, 4) for k, v in opt.items() if k != "weights"})
    print(f"QuantStats HTML report: {html_path}")
    print(f"QuantStats metrics CSV: {metrics_csv}")

    if args.export_extras:
        nav = portfolio_nav(pf_ret_0, 1.0)
        cov_a = annualized_cov(ret)
        rc = risk_contribution(w0, cov_a).sort_values(ascending=False)
        corr = corr_matrix(ret)
        var95 = var_es_hist(pf_ret_0, alpha=args.alpha)
        var99 = var_es_hist(pf_ret_0, alpha=0.99 if args.alpha < 0.99 else 0.95)

        # if benchmark exists, export excess returns and compute relative metrics
        rel_metrics = None
        if bench_ret is not None:
            idx = pf_ret_0.index.intersection(bench_ret.index)
            if len(idx) > 0:
                excess = (pf_ret_0.loc[idx] - bench_ret.loc[idx]).dropna()
                excess.to_frame('excess_return').to_csv(os.path.join(args.outdir, f"excess_returns{suffix}.csv"))
                rel_metrics = compute_relative_metrics(pf_ret_0, bench_ret)

        save_core_csvs(pf_ret_0, nav, rc, corr, args.outdir)
        plot_prices(prices, args.outdir, suffix, benchmark=bench_ret)
        os.replace(os.path.join(args.outdir, "portfolio_returns.csv"), os.path.join(args.outdir, f"portfolio_returns{suffix}.csv"))
        os.replace(os.path.join(args.outdir, "portfolio_nav.csv"), os.path.join(args.outdir, f"portfolio_nav{suffix}.csv"))
        os.replace(os.path.join(args.outdir, "risk_contribution.csv"), os.path.join(args.outdir, f"risk_contribution{suffix}.csv"))
        os.replace(os.path.join(args.outdir, "correlation_matrix.csv"), os.path.join(args.outdir, f"correlation_matrix{suffix}.csv"))

        plot_nav(nav, args.outdir)
        plot_corr(corr, args.outdir)
        os.replace(os.path.join(args.outdir, "nav.png"), os.path.join(args.outdir, f"nav{suffix}.png"))
        os.replace(os.path.join(args.outdir, "corr.png"), os.path.join(args.outdir, f"corr{suffix}.png"))

        peak, trough, mdd = max_drawdown_from_nav(nav)
        text = make_text_report(args.tickers, w0, opt.get("ann_return", float("nan")), opt.get("ann_vol", float("nan")), opt.get("sharpe", float("nan")), (peak, trough, mdd), var95, var99)
        with open(os.path.join(args.outdir, f"report{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extra CSVs, charts, and text summary exported to: {args.outdir}")

if __name__ == "__main__":
    main()
