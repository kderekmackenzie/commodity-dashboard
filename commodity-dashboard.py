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
st.write("Built by [kderekmackenzie](https://substack.com/@kderekmackenzie) — [click here](https://github.com/kderekmackenzie) for Github repo.")

st.write("Explore commodity data from 1960 to today — visualize trends, analyze seasonality, and query with OpenAI + Pinecone vector search.")


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
    percent_deviation.index = grouped.index  # Keep MultiIndex

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

# --- REMOVED THE FOLLOWING SECTION ---
# Define default years if not already defined
# years_available = sorted(data.index.year.unique())
# default_start_year = years_available[0]
# default_end_year = years_available[-1]

# Use previously defined start_year and end_year if available, else fallback to defaults
# try:
#     base_start_year
# except NameError:
#     base_start_year = default_start_year

# try:
#     base_end_year
# except NameError:
#     base_end_year = default_end_year

# Use main timeframe or fallback
# start_year = base_start_year
# end_year = base_end_year
# --- END OF REMOVED SECTION ---


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
                                    "When answering questions, prioritize information from the provided context chunks. "
                                    "If a question asks for a calculation the dashboard already performs (like seasonality), "
                                    "explain the concept, use the provided text context for specific data points if available, "
                                    "and explicitly refer to the dashboard's existing visualization/calculation capabilities "
                                    "(e.g., 'refer to the commodity seasonality chart displayed above'). "
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

