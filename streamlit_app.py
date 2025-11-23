"""
Streamlit Dashboard for Financial Statements
===========================================

This application allows you to explore historical financial statement data
stored in the provided SQLite database. You can select a company (ticker),
a statement type (income statement, balance sheet or cash flow statement)
and one or more line items to visualise. The data are loaded on demand
from the SQLite file using helper functions defined in `data_loader.py`.

Run this app locally with:

```bash
streamlit run streamlit_app.py
```

Make sure the `thegarbagebin.sqlite` file is present in the same
directory or update the `DATABASE_PATH` constant accordingly.
"""

import pathlib
from typing import List

import pandas as pd
import streamlit as st

from data_loader import (
    get_available_tickers,
    get_available_statements,
    load_statement_data,
    get_line_items,
    get_metric_series,
    get_shares_outstanding_series,
    compute_ttm,
)

# External dependency for fetching price history
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # yfinance is optional; price ratios will be unavailable


# Path to the SQLite database file. Adjust if you place the database elsewhere.
DATABASE_PATH = pathlib.Path(__file__).with_name("thegarbagebin.sqlite")


# -----------------------------------------------------------------------------
# Helper functions for the Charts & Analysis tab
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_price_history(ticker: str) -> pd.Series:
    """Load daily closing price history for a ticker using yfinance.

    The returned series is indexed by date and contains closing prices.  If
    yfinance is not installed or the download fails, an empty series is
    returned.  The results are cached by ticker to avoid repeated network
    requests.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    pandas.Series
        Series of daily closing prices.
    """
    if yf is None:
        return pd.Series(dtype=float)
    try:
        hist = yf.download(ticker, period="max", progress=False, auto_adjust=False)
        if hist.empty or "Close" not in hist:
            return pd.Series(dtype=float)
        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index, errors="coerce")
        return close.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _compute_series(
    db_path: pathlib.Path,
    ticker: str,
    statement: str,
    metric: str,
    ttm: bool,
    per_share: bool,
    price_ratio: bool,
    daily: bool,
    use_live_price: bool = False,
) -> pd.Series:
    """Compute a data series according to user selections.

    This function pulls the raw metric series from the database, optionally
    computes trailing twelve‑month values, divides by shares outstanding to
    obtain per‑share values, and divides into price to obtain price ratios.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the SQLite database file.
    ticker : str
        Ticker symbol.
    statement : str
        Statement type (e.g. "Income_Statement").
    metric : str
        Line item name from the selected statement.
    ttm : bool
        If True, compute trailing twelve‑month values.
    per_share : bool
        If True, divide the metric by shares outstanding.
    price_ratio : bool
        If True, divide the price by the per‑share metric (requires per_share).
    daily : bool
        If True and price_ratio is selected, align the per‑share metric to
        daily frequency before computing the ratio.

    Returns
    -------
    pandas.Series
        The computed series indexed by datetime.  May be empty.
    """
    # Load metric series
    series = get_metric_series(db_path, ticker, statement, metric)
    if series.empty:
        return series
    # Compute trailing twelve‑month values if requested
    if ttm:
        series = compute_ttm(series, statement)
    # Convert to per share values if requested
    if per_share or price_ratio:
        shares = get_shares_outstanding_series(db_path, ticker)
        if shares.empty:
            # If no shares series found, cannot compute per‑share values
            return pd.Series(dtype=float)
        # Align shares by forward filling the last available value
        aligned_shares = shares.reindex(series.index, method="ffill")
        per_share_series = series.div(aligned_shares)
        # Replace the working series
        series = per_share_series.dropna()
    # Compute price ratio
    if price_ratio:
        price_series = _load_price_history(ticker)
        if price_series.empty:
            return pd.Series(dtype=float)
        # If using a single live price for the ratio, divide the latest price by the per‑share values
        if use_live_price:
            # Use the most recent closing price
            try:
                latest_price = float(price_series.dropna().iloc[-1])
            except Exception:
                return pd.Series(dtype=float)
            ratio_series = pd.Series(latest_price / series, index=series.index)
            return ratio_series.dropna()
        if daily:
            # Align per‑share metric to daily frequency
            metric_daily = series.reindex(price_series.index, method="ffill")
            ratio_series = price_series.div(metric_daily)
            series = ratio_series.dropna()
        else:
            # Align price to metric frequency using backfill for quarter‑end
            aligned_price = price_series.reindex(series.index, method="bfill")
            ratio_series = aligned_price.div(series)
            series = ratio_series.dropna()
    return series


def _compute_slope_and_r2(series: pd.Series, is_daily: bool) -> tuple:
    """Compute the slope per year and R² of a linear regression for the series.

    The slope is annualised based on the frequency: for quarterly data it is
    multiplied by 4; for daily data (used when computing price ratios with
    daily prices) it is multiplied by 252 (approximate trading days per year).

    Parameters
    ----------
    series : pandas.Series
        Series of values indexed by datetime.
    is_daily : bool
        True if the series frequency is daily (price ratios with daily prices).

    Returns
    -------
    tuple
        (slope_per_year, r2).  Both values may be None if the series is
        insufficient for regression (fewer than 2 data points).
    """
    import numpy as np  # Local import to avoid unnecessary dependency at module import
    if series is None or series.empty or len(series) < 2:
        return (None, None)
    try:
        # Use numeric sequence as x to avoid issues with irregular dates
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        # Fit a first‑degree polynomial
        p = np.polyfit(x, y, 1)
        slope = p[0]
        # Compute R²
        y_pred = np.polyval(p, x)
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else None
        # Annualise slope based on frequency
        slope_per_year = None
        if slope is not None:
            slope_per_year = slope * (252 if is_daily else 4)
        return (slope_per_year, r2)
    except Exception:
        return (None, None)


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Financial Dashboard", layout="wide")
    st.title("📈 Financial Dashboard")
    st.markdown(
        """
        This app lets you explore company fundamentals from an SQLite database and
        analyse metrics with interactive charts. Use the **Financial Statements**
        tab to view raw statement data and simple charts. The **Charts & Analysis**
        tab provides a multi‑chart layout where you can compute per‑share values,
        price ratios and trailing‑twelve‑month (TTM) metrics, with slope and R²
        statistics.
        """
    )

    # Sidebar: choose ticker globally for both tabs
    st.sidebar.header("Select Company")
    tickers = get_available_tickers(DATABASE_PATH)
    selected_ticker = st.sidebar.selectbox("Ticker", sorted(tickers))

    # Tabs: Financial Statements and Charts & Analysis
    tabs = st.tabs(["Financial Statements", "Charts & Analysis"])

    # -------------------------------------------------------------------------
    # Tab 1: Financial Statements – display raw data and basic charts
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader(f"Financial Statements for {selected_ticker}")
        # Statement selection within tab
        statement_map = {
            "Income Statement": "Income_Statement",
            "Balance Sheet": "Balance_Sheet",
            "Cash Flow Statement": "Cash_Flow_Statement",
        }
        statement_label = st.selectbox(
            "Statement", list(statement_map.keys()), key="fs_statement"
        )
        selected_statement = statement_map[statement_label]
        # Line items
        items = get_line_items(DATABASE_PATH, selected_ticker, selected_statement)
        if not items:
            st.warning("No line items found for the selected statement.")
        else:
            # Default selection: Revenue if available, else first item
            default_selection: List[str] = []
            if "Revenue" in items:
                default_selection = ["Revenue"]
            elif items:
                default_selection = [items[0]]
            selected_items = st.multiselect(
                "Line Items", options=items, default=default_selection, key="fs_items"
            )
            # Load statement data once
            df = load_statement_data(DATABASE_PATH, selected_ticker, selected_statement)
            if df.empty:
                st.warning("No data found for the selected options.")
            else:
                # Melt DataFrame to long format
                df_long = (
                    df[df["Financials"].isin(selected_items)]
                    .set_index("Financials")
                    .T
                    .reset_index()
                    .rename(columns={"index": "Date"})
                )
                # Parse dates
                df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
                df_long = df_long.sort_values("Date")
                # Show data table
                st.write("### Data Table")
                st.dataframe(df_long.set_index("Date"))
                # Basic charts for each selected item
                for item in selected_items:
                    st.write(f"#### {item}")
                    chart_df = df_long[["Date", item]].set_index("Date")
                    st.line_chart(chart_df, height=250)

    # -------------------------------------------------------------------------
    # Tab 2: Charts & Analysis – multi‑chart layout with advanced metrics
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader(f"Charts & Analysis for {selected_ticker}")
        # Fetch statements and build options
        statements = get_available_statements()
        statement_labels = {
            "Income_Statement": "Income Statement",
            "Balance_Sheet": "Balance Sheet",
            "Cash_Flow_Statement": "Cash Flow Statement",
        }
        # Determine number of charts to display
        num_charts = st.number_input(
            "Number of charts", min_value=1, max_value=6, value=2, step=1, key="chart_count"
        )
        # Grid layout: 2 columns per row
        cols = st.columns(2)
        # For each chart, build UI and plot
        for chart_idx in range(num_charts):
            col = cols[chart_idx % 2]
            with col:
                st.markdown(f"**Chart {chart_idx + 1}**")
                # Statement selection for this chart
                stmt = st.selectbox(
                    "Statement",
                    options=[statement_labels[s] for s in statements],
                    key=f"chart_{chart_idx}_statement",
                )
                stmt_key = {v: k for k, v in statement_labels.items()}[stmt]
                # Line items for this statement
                items = get_line_items(DATABASE_PATH, selected_ticker, stmt_key)
                if not items:
                    st.info("No line items available.")
                    continue
                # Metric selection
                metric = st.selectbox(
                    "Metric", options=items, key=f"chart_{chart_idx}_metric"
                )
                # Options
                ttm = st.checkbox(
                    "TTM (Trailing 12M)", value=False, key=f"chart_{chart_idx}_ttm"
                )
                per_share = st.checkbox(
                    "Per Share", value=False, key=f"chart_{chart_idx}_pershare"
                )
                price_ratio = st.checkbox(
                    "Price Ratio (P/Metric)", value=False, key=f"chart_{chart_idx}_priceratio"
                )
                # Disable price ratio if per_share not selected or yfinance unavailable
                if not per_share or yf is None:
                    price_ratio = False
                use_live_price = False
                daily = False
                if price_ratio:
                    # Option to use the most recent price instead of historical prices
                    use_live_price = st.checkbox(
                        "Use Live Price", value=False, key=f"chart_{chart_idx}_use_live_price"
                    )
                    # When live price is off, allow daily vs quarterly alignment
                    if not use_live_price:
                        daily = st.checkbox(
                            "Daily Price", value=False, key=f"chart_{chart_idx}_daily"
                        )
                # Average period selection
                avg_opts = ["None"] + [f"{n}Y" for n in range(1, 8)]
                avg_period = st.selectbox(
                    "Average Window", options=avg_opts, key=f"chart_{chart_idx}_avg"
                )
                # Compute series
                series = _compute_series(
                    DATABASE_PATH,
                    selected_ticker,
                    stmt_key,
                    metric,
                    ttm,
                    per_share,
                    price_ratio,
                    daily,
                    use_live_price,
                )
                if series.empty:
                    st.warning("No data available for the selected configuration.")
                    continue
                # Compute moving average if selected
                avg_series = None
                if avg_period != "None":
                    try:
                        years = int(avg_period.replace("Y", ""))
                        window = 252 * years if (price_ratio and daily) else 4 * years
                        avg_series = series.rolling(window=window, min_periods=1).mean()
                    except Exception:
                        avg_series = None
                # Compute statistics
                slope_per_year, r2 = _compute_slope_and_r2(series, is_daily=(price_ratio and daily))
                # Prepare DataFrame for plotting
                plot_df = pd.DataFrame({"Value": series})
                if avg_series is not None:
                    plot_df["Average"] = avg_series
                # Plot using line_chart
                st.line_chart(plot_df, height=250)
                # Display statistics
                st.caption(
                    f"Latest value: {series.iloc[-1]:,.4g}"
                    + (
                        f" | Slope/yr: {slope_per_year:+.3f}"
                        if slope_per_year is not None
                        else ""
                    )
                    + (
                        f" | R²: {r2:.3f}"
                        if r2 is not None and not pd.isna(r2)
                        else ""
                    )
                )


if __name__ == "__main__":
    main()
