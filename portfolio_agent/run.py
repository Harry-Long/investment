# -*- coding: utf-8 -*-
"""
Main orchestrator for static portfolio analysis.
- Loads policy.yaml
- Resolves universe from policy (file > inline list)
- Fetches prices (Stooq or Synthetic)
- Optional pre-selection (Top by return / Sharpe)
- Optimizes weights via PyPortfolioOpt (objective from YAML)
- Optional buffet concentration enforcement (top-k min share)
- Exports QuantStats HTML and extras
"""
import os, json, argparse, pandas as pd
from mod.data_provider import get_prices_synthetic, get_prices_stooq
from mod.qs_wrapper import basic_metrics, save_html_report
from mod.risk_tools import to_simple_returns, portfolio_returns, portfolio_nav, annualized_cov, risk_contribution, corr_matrix, var_es_hist
from mod.reporting_extras import save_core_csvs, plot_nav, plot_corr, plot_prices, make_text_report, max_drawdown_from_nav, compute_relative_metrics
from mod.policy import load_policy, resolve_mode, resolve_dates_and_benchmark
from mod.universe import resolve_universe
from mod.selector import select_assets
from mod.optimizer import optimize_portfolio
from mod.backtest import Backtester
from mod.concentration import enforce_topk_share

def parse_args():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_cfg = os.path.join(base_dir, "policy.yaml")
    p = argparse.ArgumentParser(description="Portfolio analysis configured via YAML policy file")
    p.add_argument(
        "config",
        nargs="?",
        default=default_cfg,
        help=f"Path to YAML configuration file (default: {default_cfg})",
    )
    return p.parse_args()

def normalize_weights(tickers, weights):
    if weights is None:
        return pd.Series([1.0/len(tickers)]*len(tickers), index=tickers)
    if isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
        w = w.reindex(tickers).fillna(0.0)
    else:
        if len(weights) != len(tickers):
            raise ValueError("Length of weights must match tickers")
        w = pd.Series(weights, index=tickers, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be > 0")
    return w / s

def main():
    args = parse_args()
    policy = load_policy(args.config)

    data_cfg = policy.get("data") if isinstance(policy.get("data"), dict) else {}
    source = str(data_cfg.get("source", "stooq")).strip().lower() if data_cfg else "stooq"
    if source not in {"stooq", "synthetic"}:
        raise ValueError(f"Unsupported data source '{source}'. Expected 'stooq' or 'synthetic'.")

    reporting_cfg = policy.get("reporting") if isinstance(policy.get("reporting"), dict) else {}
    outdir = reporting_cfg.get("output_dir") or reporting_cfg.get("outdir") or policy.get("output_dir") or "output"
    title = reporting_cfg.get("title") or "QuantStats Report"
    var_alpha = float(reporting_cfg.get("var_alpha") or reporting_cfg.get("alpha") or 0.95)
    export_extras = bool(reporting_cfg.get("export_extras", False))

    portfolio_cfg = policy.get("portfolio") if isinstance(policy.get("portfolio"), dict) else {}
    weight_override = portfolio_cfg.get("initial_weights") if isinstance(portfolio_cfg, dict) else None

    synthetic_cfg = data_cfg.get("synthetic") if isinstance(data_cfg.get("synthetic"), dict) else {}
    synthetic_start = data_cfg.get("synthetic_start") or synthetic_cfg.get("start") or "2023-01-03"
    synthetic_periods = data_cfg.get("synthetic_periods")
    if synthetic_periods is None:
        synthetic_periods = synthetic_cfg.get("periods")
    synthetic_periods = int(synthetic_periods) if synthetic_periods is not None else (252 * 2)

    os.makedirs(outdir, exist_ok=True)
    suffix = "_synth" if source == "synthetic" else "_real"

    src = args.config if policy else "default"

    # Resolve mode
    mode_in_use, mode_note = resolve_mode(None, policy, src)
    print(f"Mode in use: {mode_in_use} ({mode_note})")

    # Resolve universe
    tickers_in_use, tickers_note = resolve_universe(policy)
    print(f"Tickers in use: {tickers_in_use} ({tickers_note})")

    # Resolve dates and benchmark
    start, end, bench_ticker = resolve_dates_and_benchmark(policy, source)

    # Fetch prices
    bench_symbol = bench_ticker.strip() if isinstance(bench_ticker, str) and bench_ticker.strip() else None

    if source == "synthetic":
        prices = get_prices_synthetic(tuple(tickers_in_use), start=synthetic_start, periods=synthetic_periods)
        bdf = None; bench_ret = None
    else:
        prices = get_prices_stooq(tuple(tickers_in_use), start=start, end=end)
        bdf = get_prices_stooq((bench_symbol,), start=start, end=end) if bench_symbol else None
        bench_ret = to_simple_returns(bdf).iloc[:, 0] if (bdf is not None and not bdf.empty) else None

    # Optional selection
    prices, chosen_list, sel_note = select_assets(prices, policy)
    print(f"Asset selection: {sel_note}")
    tickers_in_use = chosen_list

    # Returns
    ret = to_simple_returns(prices).dropna()
    if ret.empty or ret.shape[0] < 30:
        raise RuntimeError(f"Insufficient data rows: {ret.shape[0]}")

    # Initial equal weights
    w0 = normalize_weights(tickers_in_use, weight_override)
    pf_ret_0 = portfolio_returns(ret, w0)
    if pf_ret_0.empty:
        raise RuntimeError("Portfolio return series is empty.")

    # QuantStats
    metrics_equal = basic_metrics(pf_ret_0, benchmark=bench_ret)
    metrics_equal_csv = os.path.join(outdir, f"qs_metrics_equal{suffix}.csv")
    metrics_equal.to_csv(metrics_equal_csv, encoding="utf-8")
    html_equal_path = os.path.join(outdir, f"qs_report_equal{suffix}.html")
    save_html_report(pf_ret_0, html_equal_path, title=f"{title} - Equal Weights", benchmark=bench_ret)


    # Optimization
    opt = optimize_portfolio(prices, policy)
    model_name = opt.get("model", "mean_variance")
    objective_name = opt.get("objective")
    w_opt = pd.Series(opt["weights"], dtype=float).reindex(prices.columns).fillna(0.0)

    print(f"\nOptimization model: {model_name}" + (f" [{objective_name}]" if objective_name else ""))
    solver_info = opt.get("details", {})
    if isinstance(solver_info, dict) and solver_info.get("solver"):
        print(f"Optimizer backend: {solver_info['solver']}")
    guardrails_info = opt.get("guardrails") or {}
    if guardrails_info:
        print(f"Guardrails applied: {guardrails_info}")

    # Buffet concentration
    if mode_in_use == "buffet":
        port_cfg = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
        conc = (port_cfg.get("concentration") or {}) if isinstance(port_cfg, dict) else {}
        enforce_flag = conc.get("enforce")
        enforce_flag = True if enforce_flag is None else bool(enforce_flag)
        if enforce_flag:
            top_k = int(conc.get("top_k", 5))
            top_k_min_share = float(conc.get("top_k_min_share", conc.get("top5_min_share", 0.50)))
            wb = (port_cfg.get("optimization", {}) or {}).get("weight_bounds", port_cfg.get("weight_bounds", [0.0, 0.35]))
            upper_cap = float(wb[1]) if isinstance(wb, (list, tuple)) and len(wb) == 2 else 0.35
            from mod.concentration import enforce_topk_share
            w_opt = enforce_topk_share(w_opt, topk=top_k, min_share=top_k_min_share, upper=upper_cap)

    # Save weights & returns
    w0.to_frame("weight").to_csv(os.path.join(outdir, f"weights_initial{suffix}.csv"))
    w_opt.to_frame("weight").to_csv(os.path.join(outdir, f"weights_optimized{suffix}.csv"))
    pf_ret_0.to_frame("ret_initial").to_csv(os.path.join(outdir, f"returns_initial{suffix}.csv"))
    pf_ret_opt = portfolio_returns(ret, w_opt)
    pf_ret_opt.to_frame("ret_optimized").to_csv(os.path.join(outdir, f"returns_optimized{suffix}.csv"))

    # Save QuantStats for optimized portfolio
    metrics_opt = basic_metrics(pf_ret_opt, benchmark=bench_ret)
    metrics_opt_csv = os.path.join(outdir, f"qs_metrics_optimized{suffix}.csv")
    metrics_opt.to_csv(metrics_opt_csv, encoding="utf-8")
    html_opt_path = os.path.join(outdir, f"qs_report_optimized{suffix}.html")
    save_html_report(pf_ret_opt, html_opt_path, title=f"{title} - Optimized", benchmark=bench_ret)

    # --- Combined report with two strategies in one HTML ---
    from mod.reporting_extras import save_dual_report

    combined_path = os.path.join(outdir, f"qs_report_combined{suffix}.html")
    save_dual_report(
        ret_a=pf_ret_0,
        ret_b=pf_ret_opt,
        label_a="Equal Weights",
        label_b="Optimized",
        benchmark=bench_ret,
        benchmark_label=(bench_symbol or "Benchmark"),
        title=f"{title} - Combined",
        out_html=combined_path,
    )
    print(f"Combined QuantStats-like report: {combined_path}")


    print("Initial weights")
    print(w0.round(4).to_string())
    print("\nOptimized weights")
    print(w_opt.round(4).to_string())
    perf_snapshot = {
        k: (round(opt.get(k), 4) if isinstance(opt.get(k), float) else opt.get(k))
        for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")
        if opt.get(k) is not None
    }
    print("\nOptimized performance (in-sample)")
    print(perf_snapshot)
    notable_solver_fields = {}
    if isinstance(solver_info, dict):
        for key in ("risk_aversion_used", "tau", "absolute_views"):
            if key in solver_info:
                notable_solver_fields[key] = solver_info[key]
    if notable_solver_fields:
        print(f"Optimizer notes: {notable_solver_fields}")
    # print(f"QuantStats HTML report: {html_path}")
    # print(f"QuantStats metrics CSV: {metrics_csv}")

    # Backtesting
    port_section = policy.get("portfolio") if isinstance(policy.get("portfolio"), dict) else {}
    backtest_cfg = port_section.get("backtest") if isinstance(port_section, dict) else {}
    backtest_result = None
    if isinstance(backtest_cfg, dict) and backtest_cfg.get("enabled", False):
        try:
            bt = Backtester(prices, policy, returns=ret)
            backtest_result = bt.run()
        except Exception as exc:
            print(f"[warn] Backtest failed: {exc}")

    if backtest_result:
        bt_path = os.path.join(outdir, f"backtest_{backtest_result['mode']}{suffix}.json")
        with open(bt_path, "w", encoding="utf-8") as f:
            json.dump(backtest_result, f, indent=2)
        print(f"Backtest ({backtest_result['mode']}) saved to {bt_path}")

    # Extras
    if export_extras:
        nav = portfolio_nav(pf_ret_0, 1.0)
        cov_a = annualized_cov(ret)
        rc = risk_contribution(w0, cov_a).sort_values(ascending=False)
        corr = corr_matrix(ret)
        var95 = var_es_hist(pf_ret_0, alpha=var_alpha)
        var99 = var_es_hist(pf_ret_0, alpha=0.99 if var_alpha < 0.99 else 0.95)

        rel_metrics = None
        if bench_ret is not None:
            idx = pf_ret_0.index.intersection(bench_ret.index)
            if len(idx) > 0:
                excess = (pf_ret_0.loc[idx] - bench_ret.loc[idx]).dropna()
                excess.to_frame('excess_return').to_csv(os.path.join(outdir, f"excess_returns{suffix}.csv"))
                rel_metrics = compute_relative_metrics(pf_ret_0, bench_ret)

        save_core_csvs(pf_ret_0, nav, rc, corr, outdir)
        plot_prices(prices, outdir, suffix,
            benchmark=(bdf.iloc[:, 0] if bdf is not None and not bdf.empty else None),
            benchmark_label=(bench_symbol or 'Benchmark'))
        plot_nav(nav, outdir)
        plot_corr(corr, outdir)

        peak, trough, mdd = max_drawdown_from_nav(nav)
        text = make_text_report(tickers_in_use, w0,
            opt.get("ann_return", float("nan")), opt.get("ann_vol", float("nan")), opt.get("sharpe", float("nan")),
            (peak, trough, mdd), var95, var99, relative_metrics=rel_metrics)
        with open(os.path.join(outdir, f"report{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extra outputs saved to {outdir}")

if __name__ == "__main__":
    main()
