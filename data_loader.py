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
