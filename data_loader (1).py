"""
Utility functions for accessing financial statement data stored in an SQLite
database. This module encapsulates the database queries used by the
Streamlit dashboard so that they can be reused and unit tested
independently of the Streamlit UI.
"""

import sqlite3
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Create a read-only connection to the SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        A SQLite connection object configured for read-only access.
    """
    # SQLite allows URI connections; using uri=True ensures that connecting to
    # the database with `mode=ro` opens the file in read-only mode.
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def get_available_tickers(db_path: Path) -> List[str]:
    """Return a list of all unique ticker symbols in the database.

    Each table in the database follows the naming pattern
    `<TICKER>_<STATEMENT_TYPE>`, e.g. `ABG_Income_Statement`.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    List[str]
        Sorted list of ticker symbols.
    """
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row[0] for row in cursor.fetchall()]
    tickers = {
        name.split("_")[0] for name in table_names if "_" in name
    }
    return sorted(tickers)


def get_available_statements() -> List[str]:
    """Return the list of supported statement types.

    These correspond to the suffixes used in the table names.
    """
    return ["Income_Statement", "Balance_Sheet", "Cash_Flow_Statement"]


def get_line_items(db_path: Path, ticker: str, statement: str) -> List[str]:
    """Return the list of line items available for a given ticker and statement.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    ticker : str
        Ticker symbol (e.g. "ABG").
    statement : str
        Statement type (one of the values returned by
        :func:`get_available_statements`).

    Returns
    -------
    List[str]
        Sorted list of line item names.
    """
    table_name = f"{ticker}_{statement}"
    with _get_connection(db_path) as conn:
        try:
            df = pd.read_sql_query(f"SELECT Financials FROM {table_name}", conn)
        except Exception:
            return []
    return sorted(df["Financials"].dropna().unique().tolist())


def load_statement_data(db_path: Path, ticker: str, statement: str) -> pd.DataFrame:
    """Load the full statement data for a given ticker and statement type.

    The returned DataFrame has a column ``Financials`` listing the line
    items and additional columns for each date.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    ticker : str
        Ticker symbol.
    statement : str
        Statement type (e.g. "Income_Statement").

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the statement data, or an empty
        DataFrame if the table does not exist.
    """
    table_name = f"{ticker}_{statement}"
    with _get_connection(db_path) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            # Table not found or other error
            return pd.DataFrame()
    return df

def get_metric_series(db_path: Path, ticker: str, statement: str, metric: str) -> pd.Series:
    """Return a quarterly series for a specific line item (metric).

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    ticker : str
        Ticker symbol.
    statement : str
        Statement type (e.g. "Income_Statement").
    metric : str
        Name of the line item/metric to return.

    Returns
    -------
    pandas.Series
        A Series indexed by datetime with the metric values as floats.  If the
        metric or table is not found, an empty Series is returned.
    """
    df = load_statement_data(db_path, ticker, statement)
    if df.empty:
        return pd.Series(dtype=float)
    if metric not in df["Financials"].values:
        return pd.Series(dtype=float)
    row = df[df["Financials"] == metric]
    # Drop the Financials column and transpose so dates become the index
    series = row.drop(columns=["Financials"]).T
    series.index.name = "Date"
    series.columns = [metric]
    # Convert the index to datetime and coerce numeric values
    try:
        series.index = pd.to_datetime(series.index, errors="coerce")
    except Exception:
        series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[metric].apply(pd.to_numeric, errors="coerce")
    # Drop missing dates and values
    series = series.dropna()
    return series.sort_index()

# Keywords to identify share count metrics in a statement
_SHARES_METRIC_KEYWORDS = [
    "diluted weighted average shares outstanding",
    "weighted average shares outstanding diluted",
    "shares outstanding diluted",
    "basic weighted average shares outstanding",
    "weighted average shares outstanding basic",
    "shares outstanding basic",
    "shares outstanding",
]


def get_shares_outstanding_series(db_path: Path, ticker: str) -> pd.Series:
    """Search for and return a shares outstanding series for the given ticker.

    This helper looks for a line item in the income statement that matches
    common share count keywords.  If a matching item is found, its series
    is returned.  Otherwise, an empty Series is returned.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    ticker : str
        Ticker symbol.

    Returns
    -------
    pandas.Series
        A Series indexed by datetime with the share counts, or an empty
        Series if no suitable metric is found.
    """
    # Look through the income statement first for share count metrics
    statement = "Income_Statement"
    items = get_line_items(db_path, ticker, statement)
    # Build a mapping of lowercase names to original names for matching
    lookup = {name.lower().strip(): name for name in items}
    for keyword in _SHARES_METRIC_KEYWORDS:
        key = keyword.lower().strip()
        if key in lookup:
            metric_name = lookup[key]
            return get_metric_series(db_path, ticker, statement, metric_name)
    # Fallback: search in balance sheet (some databases store shares there)
    statement = "Balance_Sheet"
    items = get_line_items(db_path, ticker, statement)
    lookup = {name.lower().strip(): name for name in items}
    for keyword in _SHARES_METRIC_KEYWORDS:
        key = keyword.lower().strip()
        if key in lookup:
            metric_name = lookup[key]
            return get_metric_series(db_path, ticker, statement, metric_name)
    return pd.Series(dtype=float)


def compute_ttm(series: pd.Series, statement: str) -> pd.Series:
    """Compute a trailing twelve‑month (TTM) series from a quarterly series.

    For balance sheet items, the TTM is calculated as the rolling average of
    the last four periods.  For income/cash flow items, the TTM is the rolling
    sum of the last four periods.

    Parameters
    ----------
    series : pandas.Series
        Quarterly series indexed by datetime.
    statement : str
        Name of the statement (e.g. "Income_Statement", "Balance_Sheet").

    Returns
    -------
    pandas.Series
        A Series representing the TTM values.  Empty if insufficient data.
    """
    if series.empty:
        return series
    # Sort by date to ensure chronological order
    series = series.sort_index()
    if "Balance" in statement:
        # Rolling mean of last 4 quarters
        ttm_series = series.rolling(window=4, min_periods=1).mean()
    else:
        # Rolling sum of last 4 quarters; require 4 periods to compute
        ttm_series = series.rolling(window=4, min_periods=4).sum()
    # Drop NaNs introduced by rolling
    return ttm_series.dropna()
