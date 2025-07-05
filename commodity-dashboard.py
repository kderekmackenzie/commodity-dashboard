import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load historical commodity price data
data = pd.read_csv(
    "commodity-dashboard-prices.csv",
    parse_dates=['Date'],
    na_values=["...", "…"]
)
data.set_index('Date', inplace=True)

# Clean all columns: remove commas, trim whitespace, handle unicode ellipsis
for col in data.columns:
    data[col] = (
        data[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("…", "", regex=False)
        .str.strip()
    )
    data[col] = pd.to_numeric(data[col], errors='coerce')

st.title("📈 What If I Bought Commodities?")
st.write("Select a year and an investment amount to see what your commodity investment would be worth today.")

# Sidebar inputs
start_year = st.sidebar.selectbox("Start Year", sorted(data.index.year.unique()))
amount_invested = st.sidebar.number_input("Investment Amount ($)", min_value=100.0, value=1000.0, step=100.0)

# Safe default commodities (only if they exist in the dataset)
default_commodities = [
    c for c in [
        'Gold ($/troy oz)', 
        'Silver ($/troy oz)', 
        'Copper ($/mt)', 
        'Crude oil, average ($/bbl)', 
        'S&P500'
    ] if c in data.columns
]
commodities = st.sidebar.multiselect("Choose Commodities", options=data.columns.tolist(), default=default_commodities)

# Let user choose chart mode
chart_mode = st.sidebar.radio("Chart Mode", ["Raw Prices", "Normalized to 100", "Percent Change"])

# Find first available month of the selected year
start_date = data[data.index.year == start_year].index.min()

if start_date is None or not commodities:
    st.warning("Start date or valid commodity selection not available.")
else:
    # Extract and coerce prices
    start_prices = pd.to_numeric(data.loc[start_date, commodities], errors='coerce')
    end_prices = pd.to_numeric(data.iloc[-1][commodities], errors='coerce')

    # Filter out invalid entries
    valid_commodities = start_prices.dropna().index.intersection(end_prices.dropna().index)

    if valid_commodities.empty:
        st.error("All selected commodities have missing price data. Try a different year or selection.")
        st.write("Missing start prices for:", start_prices[start_prices.isna()].index.tolist())
        st.write("Missing end prices for:", end_prices[end_prices.isna()].index.tolist())
        st.stop()

    start_prices = start_prices[valid_commodities]
    end_prices = end_prices[valid_commodities]

    # Calculate results
    years = (data.index[-1] - pd.to_datetime(start_date)).days / 365.25
    units_purchased = amount_invested / start_prices
    current_value = units_purchased * end_prices
    returns = ((end_prices - start_prices) / start_prices) * 100
    cagr = ((end_prices / start_prices) ** (1 / years) - 1) * 100

    summary = pd.DataFrame({
        'Start Price': start_prices.round(2),
        'End Price': end_prices.round(2),
        'Units Bought': units_purchased.round(2),
        'Value Today ($)': current_value.round(2),
        'Total Return (%)': returns.round(2),
        'CAGR (%)': cagr.round(2)
    }).sort_values("CAGR (%)", ascending=False)

    st.subheader(f"Investment Outcome Since {start_year} (Starting with ${amount_invested:.2f})")
    st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen'))

    # Price chart
    st.subheader("📉 Price History")
    if chart_mode == "Raw Prices":
        st.line_chart(data[valid_commodities])
    elif chart_mode == "Normalized to 100":
        normalized = data[valid_commodities] / data[valid_commodities].iloc[0] * 100
        st.line_chart(normalized)
    elif chart_mode == "Percent Change":
        start_vals = data.loc[start_date, valid_commodities]
        pct_change = data[valid_commodities].divide(start_vals) * 100
        st.line_chart(pct_change)

    # Correlation heatmap
    st.subheader("🔍 Commodity Price Correlation Heatmap")
    corr = data[valid_commodities].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
