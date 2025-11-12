import streamlit as st
import pandas as pd
from src.data_loader import load_country_data, generate_boxplot, top_regions

st.set_page_config(page_title="Solar Insights Dashboard", layout="wide")

st.title("🌞 Solar Energy Insights Dashboard")

# Sidebar widgets
st.sidebar.header("Filter Options")
country = st.sidebar.selectbox(
    "Select Country", ["Sierra Leone", "Togo", "Benin"])
value_column = st.sidebar.selectbox(
    "Select Variable", ["GHI", "Temperature", "Irradiance"])

# Map country to CSV
country_files = {
    "Sierra Leone": "data/sierralione_clean.csv",
    "Togo": "data/togo_clean.csv",
    "Benin": "data/benin_clean.csv"
}

# Load data
df = load_country_data(country_files[country])

st.subheader(f"Boxplot of {value_column} in {country}")
plt = generate_boxplot(df, value_column)
st.pyplot(plt)

st.subheader(f"Top Regions by Average {value_column}")
top_df = top_regions(df, value_column)
st.table(top_df)
