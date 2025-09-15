import quantstats as qs
import pandas as pd

def basic_metrics(ret: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    return qs.reports.metrics(ret, benchmark=benchmark, display=False, mode='basic')

def save_html_report(ret: pd.Series, out_html: str, title: str = 'QuantStats Report', benchmark: pd.Series | None = None):
    qs.reports.html(ret, benchmark, output=out_html, title=title)
    return out_html
