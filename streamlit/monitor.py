import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px
import os
from dotenv import load_dotenv

# Load .env for DagsHub credentials
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Page settings
st.set_page_config(page_title="Citi Bike Monitoring Dashboard", layout="wide")

# App header
st.title("🚲 Citi Bike Model Monitoring")
st.subheader("Track Performance and Predictions Across Stations")

# Sidebar navigation
page = st.sidebar.radio("📌 Choose a View", ["📉 Model MAE Comparison", "📈 Prediction Viewer"])

if page == "📉 Model MAE Comparison":
    st.markdown("### 📊 MAE Across Trained Models")

    # Load run data from MLflow
    df = mlflow.search_runs(experiment_ids=["0"], filter_string="", output_format="pandas")

    # Safe column handling
    if "tags.mlflow.runName" in df.columns:
        df = df[["tags.mlflow.runName", "metrics.mae"]]
        df.columns = ["Model", "MAE"]
    else:
        df = df[["metrics.mae"]]
        df["Model"] = [f"Run {i+1}" for i in range(len(df))]
        df = df[["Model", "metrics.mae"]]
        df.columns = ["Model", "MAE"]

    df = df.sort_values("MAE")
    col1, col2 = st.columns(2)
    col1.metric("📉 Best MAE", f"{df['MAE'].min():.2f}")
    col2.metric("📈 Worst MAE", f"{df['MAE'].max():.2f}")

    st.dataframe(df.reset_index(drop=True))

    # Styled bar chart
    fig = px.bar(df, x="Model", y="MAE", title="Model Comparison: MAE", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Prediction Viewer":
    st.markdown("### 🔍 Predicted Ride Counts by Station")

    try:
        pred_df = pd.read_csv("./data/predictions/predictions.csv", parse_dates=["datetime"])
        pred_df["start_station_id"] = pred_df["start_station_id"].astype(str)

        station_options = sorted(pred_df["start_station_id"].unique())
        selected_station = st.selectbox("Select a Station", station_options)

        filtered_df = pred_df[pred_df["start_station_id"] == selected_station]

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("📊 Total Predicted Rides", f"{int(filtered_df['predicted_ride_count'].sum()):,}")
        col2.metric("🚀 Peak Prediction", f"{int(filtered_df['predicted_ride_count'].max())}")

        # Styled line chart
        fig = px.line(
            filtered_df,
            x="datetime",
            y="predicted_ride_count",
            title=f"Hourly Predicted Rides - Station {selected_station}",
            markers=True,
            labels={"predicted_ride_count": "Predicted Rides"},
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Prediction file not found. Please run inference first.")
