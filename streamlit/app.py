import streamlit as st
import pandas as pd
import plotly.express as px

# Load predictions
df = pd.read_csv("./data/predictions/predictions.csv", parse_dates=["datetime"])

# Page config
st.set_page_config(page_title="Citi Bike Forecasting", layout="wide")

# Title and subheader
st.title("🚲 Citi Bike Forecasting")
st.subheader("Hourly Ride Predictions by Station (Interactive Dashboard)")

# Sidebar - Station selection
station_ids = sorted(df["start_station_id"].unique())
selected_station = st.sidebar.selectbox("Select a Station", station_ids)

# Filter data by selected station
station_df = df[df["start_station_id"] == selected_station]

# Top-level metrics
col1, col2 = st.columns(2)
col1.metric("📊 Total Predicted Rides", f"{int(station_df['predicted_ride_count'].sum()):,}")
col2.metric("🚀 Peak Prediction", f"{int(station_df['predicted_ride_count'].max())}")

# Line plot with dark theme
fig = px.line(
    station_df,
    x="datetime",
    y="predicted_ride_count",
    title=f"📈 Predicted Hourly Rides - Station {selected_station}",
    markers=True,
    labels={"predicted_ride_count": "Predicted Rides"},
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Expandable raw data table
with st.expander("📋 View Prediction Data"):
    st.dataframe(station_df.set_index("datetime"))
