# -*- coding: utf-8 -*-
"""
Main orchestrator for static portfolio analysis.
- Loads policy.yaml
- Resolves universe (CLI > file > inline)
- Fetches prices (Stooq or Synthetic)
- Optional pre-selection (Top by return / Sharpe)
- Optimizes weights via PyPortfolioOpt (objective from YAML)
- Optional buffet concentration enforcement (top-k min share)
- Exports QuantStats HTML and extras
"""
import os, argparse, pandas as pd, numpy as np
from mod.data_provider import get_prices_synthetic, get_prices_stooq
from mod.qs_wrapper import basic_metrics, save_html_report
from mod.risk_tools import to_simple_returns, portfolio_returns, portfolio_nav, annualized_cov, risk_contribution, corr_matrix, var_es_hist
from mod.reporting_extras import save_core_csvs, plot_nav, plot_corr, plot_prices, make_text_report, max_drawdown_from_nav, compute_relative_metrics
from mod.policy import load_policy, resolve_mode, resolve_dates_and_benchmark
from mod.universe import resolve_universe
from mod.selector import select_assets
from mod.optimizer import optimize_portfolio
from mod.concentration import enforce_topk_share

def parse_args():
    p = argparse.ArgumentParser(description="Portfolio analysis with QuantStats + PyPortfolioOpt (Stooq backend only)")
    p.add_argument("--source", choices=["synthetic","stooq"], default="synthetic")
    p.add_argument("--tickers", nargs="+", default=None, help="Override asset universe at runtime")
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--synthetic-start", type=str, default="2023-01-03")
    p.add_argument("--synthetic-periods", type=int, default=252*2)
    p.add_argument("--outdir", type=str, default="output")
    p.add_argument("--title", type=str, default="QuantStats Report")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--config", type=str, default="policy.yaml", help="Policy YAML file")
    p.add_argument("--mode", choices=["naive","buffet"], default=None, help="Override mode at runtime; does not modify YAML")
    p.add_argument("--benchmark", type=str, default="SPY.US")
    p.add_argument("--export-extras", action="store_true")
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

    # Load policy
    policy = load_policy(args.config)
    src = args.config if policy else "default"

    # Resolve mode
    mode_in_use, mode_note = resolve_mode(args.mode, policy, src)
    print(f"Mode in use: {mode_in_use} ({mode_note})")

    # Resolve universe
    tickers_in_use, tickers_note = resolve_universe(args.tickers, policy)
    print(f"Tickers in use: {tickers_in_use} ({tickers_note})")

    # Resolve dates and benchmark
    start, end, bench_ticker = resolve_dates_and_benchmark(args, policy)

    # Fetch prices
    if args.source == "synthetic":
        prices = get_prices_synthetic(tuple(tickers_in_use), start=args.synthetic_start, periods=args.synthetic_periods)
        bdf = None; bench_ret = None
    else:
        prices = get_prices_stooq(tuple(tickers_in_use), start=start, end=end)
        bdf = get_prices_stooq((bench_ticker.strip(),), start=start, end=end) if bench_ticker else None
        bench_ret = to_simple_returns(bdf).iloc[:, 0] if (bdf is not None and not bdf.empty) else None

    # Optional selection
    prices, chosen_list, sel_note = select_assets(prices, policy)
    print(f"Asset selection: {sel_note}")
    tickers_in_use = chosen_list

    # Returns
    ret = to_simple_returns(prices)
    if ret.empty or ret.shape[0] < 30:
        raise RuntimeError(f"Insufficient data rows: {ret.shape[0]}")

    # Initial equal weights
    w0 = normalize_weights(tickers_in_use, args.weights)
    pf_ret_0 = portfolio_returns(ret, w0)
    if pf_ret_0.empty:
        raise RuntimeError("Portfolio return series is empty.")

    # QuantStats
    metrics_equal = basic_metrics(pf_ret_0, benchmark=bench_ret)
    metrics_equal_csv = os.path.join(args.outdir, f"qs_metrics_equal{suffix}.csv")
    metrics_equal.to_csv(metrics_equal_csv, encoding="utf-8")
    html_equal_path = os.path.join(args.outdir, f"qs_report_equal{suffix}.html")
    save_html_report(pf_ret_0, html_equal_path, title=f"{args.title} - Equal Weights", benchmark=bench_ret)


    # Optimization
    opt = optimize_portfolio(prices, policy)
    w_opt = pd.Series(opt["weights"]).reindex(prices.columns).fillna(0.0)

    # Buffet concentration
    if mode_in_use == "buffet":
        port_cfg = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
        conc = (port_cfg.get("concentration") or {}) if isinstance(port_cfg, dict) else {}
        top_k = int(conc.get("top_k", 5))
        top_k_min_share = float(conc.get("top_k_min_share", conc.get("top5_min_share", 0.50)))
        wb = (port_cfg.get("optimization", {}) or {}).get("weight_bounds", port_cfg.get("weight_bounds", [0.0, 0.35]))
        upper_cap = float(wb[1]) if isinstance(wb, (list, tuple)) and len(wb) == 2 else 0.35
        from mod.concentration import enforce_topk_share
        w_opt = enforce_topk_share(w_opt, topk=top_k, min_share=top_k_min_share, upper=upper_cap)

    # Save weights & returns
    w0.to_frame("weight").to_csv(os.path.join(args.outdir, f"weights_initial{suffix}.csv"))
    w_opt.to_frame("weight").to_csv(os.path.join(args.outdir, f"weights_optimized{suffix}.csv"))
    pf_ret_0.to_frame("ret_initial").to_csv(os.path.join(args.outdir, f"returns_initial{suffix}.csv"))
    pf_ret_opt = portfolio_returns(ret, w_opt)
    pf_ret_opt.to_frame("ret_optimized").to_csv(os.path.join(args.outdir, f"returns_optimized{suffix}.csv"))

    # Save QuantStats for optimized portfolio
    metrics_opt = basic_metrics(pf_ret_opt, benchmark=bench_ret)
    metrics_opt_csv = os.path.join(args.outdir, f"qs_metrics_optimized{suffix}.csv")
    metrics_opt.to_csv(metrics_opt_csv, encoding="utf-8")
    html_opt_path = os.path.join(args.outdir, f"qs_report_optimized{suffix}.html")
    save_html_report(pf_ret_opt, html_opt_path, title=f"{args.title} - Optimized", benchmark=bench_ret)

    # --- Combined report with two strategies in one HTML ---
    from mod.reporting_extras import save_dual_report

    combined_path = os.path.join(args.outdir, f"qs_report_combined{suffix}.html")
    save_dual_report(
        ret_a=pf_ret_0,
        ret_b=pf_ret_opt,
        label_a="Equal Weights",
        label_b="Optimized",
        benchmark=bench_ret,
        benchmark_label=(bench_ticker if bench_ticker else "Benchmark"),
        title=f"{args.title} - Combined",
        out_html=combined_path,
    )
    print(f"Combined QuantStats-like report: {combined_path}")


    print("Initial weights")
    print(w0.round(4).to_string())
    print("\nOptimized weights (PyPortfolioOpt)")
    print(w_opt.round(4).to_string())
    print("\nOptimized performance (estimate)")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in opt.items() if k != "weights"})
    # print(f"QuantStats HTML report: {html_path}")
    # print(f"QuantStats metrics CSV: {metrics_csv}")

    # Extras
    if args.export_extras:
        nav = portfolio_nav(pf_ret_0, 1.0)
        cov_a = annualized_cov(ret)
        rc = risk_contribution(w0, cov_a).sort_values(ascending=False)
        corr = corr_matrix(ret)
        var95 = var_es_hist(pf_ret_0, alpha=args.alpha)
        var99 = var_es_hist(pf_ret_0, alpha=0.99 if args.alpha < 0.99 else 0.95)

        rel_metrics = None
        if bench_ret is not None:
            idx = pf_ret_0.index.intersection(bench_ret.index)
            if len(idx) > 0:
                excess = (pf_ret_0.loc[idx] - bench_ret.loc[idx]).dropna()
                excess.to_frame('excess_return').to_csv(os.path.join(args.outdir, f"excess_returns{suffix}.csv"))
                rel_metrics = compute_relative_metrics(pf_ret_0, bench_ret)

        save_core_csvs(pf_ret_0, nav, rc, corr, args.outdir)
        plot_prices(prices, args.outdir, suffix,
            benchmark=(bdf.iloc[:, 0] if bdf is not None and not bdf.empty else None),
            benchmark_label=(bench_ticker if bench_ticker else 'Benchmark'))
        plot_nav(nav, args.outdir)
        plot_corr(corr, args.outdir)

        peak, trough, mdd = max_drawdown_from_nav(nav)
        text = make_text_report(tickers_in_use, w0,
            opt.get("ann_return", float("nan")), opt.get("ann_vol", float("nan")), opt.get("sharpe", float("nan")),
            (peak, trough, mdd), var95, var99, relative_metrics=rel_metrics)
        with open(os.path.join(args.outdir, f"report{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extra outputs saved to {args.outdir}")

if __name__ == "__main__":
    main()
