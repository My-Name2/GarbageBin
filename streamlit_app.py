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
)


# Path to the SQLite database file. Adjust if you place the database elsewhere.
DATABASE_PATH = pathlib.Path(__file__).with_name("thegarbagebin.sqlite")


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Financial Statements Dashboard", layout="wide")
    st.title("📊 Financial Statements Dashboard")
    st.markdown(
        """
        Use the sidebar to select a company ticker, statement type and
        line items to visualise. The charts update automatically once you
        make your selections. Data come from an SQLite database included
        with this project.
        """
    )

    # Sidebar selections
    st.sidebar.header("Options")
    # Load tickers from database
    tickers = get_available_tickers(DATABASE_PATH)
    selected_ticker = st.sidebar.selectbox("Select company (ticker)", sorted(tickers))
    statements = get_available_statements()
    statement_map = {
        "Income Statement": "Income_Statement",
        "Balance Sheet": "Balance_Sheet",
        "Cash Flow Statement": "Cash_Flow_Statement",
    }
    selected_statement_label = st.sidebar.selectbox("Select statement type", list(statement_map.keys()))
    selected_statement = statement_map[selected_statement_label]

    # Load list of available line items for the selected table
    line_items = get_line_items(DATABASE_PATH, selected_ticker, selected_statement)
    # Multi-select line items to plot
    default_selection: List[str] = []
    if "Revenue" in line_items:
        default_selection.append("Revenue")
    elif line_items:
        default_selection.append(line_items[0])
    selected_items = st.sidebar.multiselect(
        "Select line items", options=line_items, default=default_selection
    )

    # Load data for the selected ticker and statement
    df = load_statement_data(DATABASE_PATH, selected_ticker, selected_statement)
    if df.empty:
        st.warning("No data found for the selected options.")
        return

    # Melt the dataframe so that each line item becomes a column with date/value pairs
    df_long = (
        df[df["Financials"].isin(selected_items)]
        .set_index("Financials")
        .T
        .reset_index()
        .rename(columns={"index": "Date"})
    )
    # Convert Date column to datetime to enable proper sorting and charting
    try:
        df_long["Date"] = pd.to_datetime(df_long["Date"])
    except Exception:
        # Some statements use month-end dates (e.g. 2011-01-31); letting
        # pandas parse them automatically yields correct datetime values
        df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")
    df_long = df_long.sort_values("Date")

    # Display selected data in table form
    st.subheader("Raw Data")
    st.dataframe(df_long.set_index("Date"))

    # Plot each selected line item
    st.subheader("Charts")
    for item in selected_items:
        st.write(f"### {item}")
        chart_df = df_long[["Date", item]].set_index("Date")
        st.line_chart(chart_df, height=300)


if __name__ == "__main__":
    main()