# commodity-dashboard.py 

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from pinecone_setup import get_pinecone_index, query_pinecone_index

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

st.title("Commodity Dashboard")
st.write("Built by [kderekmackenzie](https://substack.com/@kderekmackenzie) — for Github repo [click here](https://github.com/kderekmackenzie/commodity-dashboard).")

st.write("Explore commodity data from 1960 to today — visualize trends, analyze seasonality, relative strength and query with OpenAI + Pinecone vector search.")


st.write("Select a year and an investment amount to see what a commodity investment would be worth today.")


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

# Options for the relative strength benchmark
# Ensure these exist in your data.columns
relative_strength_benchmark_options = [
    c for c in ['S&P500', 'Gold ($/troy oz)', 'Silver ($/troy oz)', 'Copper ($/mt)', 'Crude oil, average ($/bbl)']
    if c in data.columns
]
# Set default to S&P500 if available, otherwise the first option
default_rs_benchmark = 'S&P500' if 'S&P500' in relative_strength_benchmark_options else (relative_strength_benchmark_options[0] if relative_strength_benchmark_options else None)

relative_strength_benchmark = None
if relative_strength_benchmark_options:
    relative_strength_benchmark = st.sidebar.selectbox(
        "Compare Relative Strength Against",
        options=relative_strength_benchmark_options,
        index=relative_strength_benchmark_options.index(default_rs_benchmark) if default_rs_benchmark else 0
    )
else:
    st.sidebar.info("No valid commodities available for relative strength comparison.")


# Let user choose chart mode
chart_mode = st.sidebar.radio("Chart Mode", ["Normalized to 100", "Percent Change", "Raw Prices" ])

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
    filtered_data = data[data.index >= start_date][valid_commodities]

    if chart_mode == "Raw Prices":
        st.line_chart(filtered_data)
    elif chart_mode == "Normalized to 100":
        normalized = filtered_data / filtered_data.iloc[0] * 100
        st.line_chart(normalized)
    elif chart_mode == "Percent Change":
        start_vals = filtered_data.iloc[0]
        pct_change = filtered_data.divide(start_vals) * 100
        st.line_chart(pct_change)

    # Correlation heatmap
    st.subheader("🔍 Commodity Price Correlation Heatmap")
    corr = filtered_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    # Seasonality analysis with percent deviation from yearly mean
    st.subheader("📅 Seasonal Averages by Month (Deviation from Yearly Mean)")
    seasonal_data = data[valid_commodities].copy()
    seasonal_data['Year'] = seasonal_data.index.year
    seasonal_data['Month'] = seasonal_data.index.month

    # Group and normalize as percent deviation from the yearly mean
    grouped = seasonal_data.groupby(['Year', 'Month']).mean()
    yearly_means = grouped.groupby(level=0).transform('mean')
    percent_deviation = ((grouped - yearly_means) / yearly_means) * 100
    percent_deviation.index = grouped.index # Keep MultiIndex

    # Average across years per month
    monthly_deviation = percent_deviation.groupby(level='Month').mean()

    for commodity in valid_commodities:
        fig3, ax3 = plt.subplots()
        monthly_deviation[commodity].plot(ax=ax3)
        ax3.set_title(f"Seasonality for {commodity} (% Deviation from Yearly Mean)")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("% Deviation")
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.axhline(0, linestyle='--', color='gray', linewidth=1)
        st.pyplot(fig3)

    # Relative Strength Charts
    st.subheader(f"💪 Relative Strength vs. {relative_strength_benchmark}")
    benchmark_column_name = relative_strength_benchmark

    # Check if the selected benchmark data is available and valid
    if benchmark_column_name is None or benchmark_column_name not in data.columns or data[benchmark_column_name].isna().all():
        st.warning(f"{benchmark_column_name} data is not available or is entirely missing. Cannot calculate relative strength.")
    else:
        # Filter for data from the selected start_date onwards
        relative_strength_data = data[data.index >= start_date].copy()

        # Ensure benchmark column is numerical and non-zero
        benchmark_prices = pd.to_numeric(relative_strength_data[benchmark_column_name], errors='coerce').dropna()

        if benchmark_prices.empty or (benchmark_prices == 0).any():
            st.warning(f"{benchmark_column_name} data is invalid or contains zero values, preventing relative strength calculation.")
        else:
            # Exclude the benchmark commodity itself from the list of commodities to compare
            commodities_for_rs = [c for c in valid_commodities if c != benchmark_column_name]

            if not commodities_for_rs:
                st.info(f"Select commodities (other than {benchmark_column_name}) to view their relative strength.")
            else:
                for commodity in commodities_for_rs:
                    commodity_prices = pd.to_numeric(relative_strength_data[commodity], errors='coerce').dropna()

                    # Align indices and drop NaNs for consistent calculation
                    aligned_data = pd.DataFrame({
                        'Commodity': commodity_prices,
                        benchmark_column_name: benchmark_prices
                    }).dropna()

                    if not aligned_data.empty:
                        # Calculate relative strength: Commodity Price / Benchmark Price
                        # Handle potential division by zero by replacing 0s in benchmark_prices with NaN before division
                        aligned_data[f'{benchmark_column_name}_clean'] = aligned_data[benchmark_column_name].replace(0, pd.NA)
                        relative_strength = (aligned_data['Commodity'] / aligned_data[f'{benchmark_column_name}_clean']).dropna()

                        if not relative_strength.empty:
                            fig_rs, ax_rs = plt.subplots(figsize=(10, 5))
                            ax_rs.plot(relative_strength.index, relative_strength.values)
                            ax_rs.set_title(f"Relative Strength of {commodity} vs. {benchmark_column_name}")
                            ax_rs.set_xlabel("Date")
                            ax_rs.set_ylabel(f"Ratio ({commodity} / {benchmark_column_name})")
                            ax_rs.grid(True)
                            st.pyplot(fig_rs)
                        else:
                            st.info(f"Not enough common data for {commodity} and {benchmark_column_name} to calculate relative strength.")
                    else:
                        st.info(f"Not enough common data for {commodity} and {benchmark_column_name} to calculate relative strength.")



# OpenAI Q&A section
st.subheader("Ask Questions About the Dataset")
api_key = st.text_input("Enter your OpenAI API key to enable natural language queries:", type="password")

if api_key:
    user_query = st.text_area("Ask a question about the data")

    if st.button("Submit Query") and user_query:
        with st.spinner("Thinking..."): # This line was incorrectly indented in the original input

            # Retrieve relevant context from ChromaDB
            try:
                context_chunks = query_pinecone_index(api_key, user_query, n_results=30)
                context = "\n\n".join(context_chunks).strip()

                if not context:
                    st.warning("No relevant data found in the vector database for this question.")
                else:
                    # Construct prompt
                    prompt = f"""
You are a data analyst. Use the following data to answer the user's question.

Context:
{context}

Question:
{user_query}

Answer:
"""

                    # Query GPT
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            # --- MODIFIED SYSTEM MESSAGE HERE ---
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert data analyst assistant for a Streamlit commodity dashboard. "
                                    "This dashboard processes a comprehensive historical commodity price dataset and "
                                    "can calculate and visualize metrics like correlation, returns, and seasonality. "
                                    "**Your primary role is to answer the user's questions directly and factually using the provided context chunks.** "
                                    "You are capable of interpreting the numerical data within the text context to derive specific answers (e.g., calculating rates of return, identifying best performers). "
                                    "If the dashboard also visualizes this information, you may, *after providing your direct answer*, optionally refer the user to the relevant chart for further details or verification. "
                                    "Do not suggest external tools for tasks the dashboard already handles. "
                                    "Keep your answers concise and to the point."
                                    "Do not state that data is missing or suggest external tools for tasks the dashboard already handles."
                                    "Do always provide an answer derived from the data and some context that would be useful to someone curious about this dataset."
                                )
                            },
                            # --- END MODIFIED SYSTEM MESSAGE ---
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.success(answer)

            except Exception as e:
                st.error(f"Error: {e}")

# Share raw data 


st.write("View the raw data source from World Bank:")
st.write("[Download Monthly Prices XLS - Current to July 2025](https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/related/CMO-Historical-Data-Monthly.xlsx)")

