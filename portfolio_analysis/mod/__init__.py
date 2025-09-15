# mod/__init__.py
from .data_provider import get_prices_synthetic, get_prices_stooq
from .qs_wrapper import basic_metrics, save_html_report
from .pypfopt_wrapper import optimize_max_sharpe
from .risk_tools import (
    to_simple_returns,
    portfolio_returns,
    portfolio_nav,
    annualized_cov,
    risk_contribution,
    corr_matrix,
    var_es_hist,
)
from .reporting_extras import (
    save_core_csvs,
    plot_nav,
    plot_corr,
    plot_prices,
    compute_relative_metrics,
    plot_prices,
    make_text_report,
    max_drawdown_from_nav,
)
