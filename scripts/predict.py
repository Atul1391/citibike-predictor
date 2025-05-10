import pandas as pd
import mlflow
from dotenv import load_dotenv
import os

# Step 1: Load environment variables
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("citibike_trip_prediction")

# Step 2: Load your feature dataset
df = pd.read_csv("./data/features/citibike_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Step 3: Select recent time window for inference
recent_df = df.groupby("start_station_id").tail(48)

# ✅ Step 4: Use the exact 10 features used in training
trained_features = [
    'rolling_mean_24', 'hour', 'rolling_std_6', 'rolling_mean_12', 
    'day', 'rolling_mean_6', 'lag_1', 'weekday', 'month', 'lag_21'
]
X_recent = recent_df[trained_features]

# Step 5: Load best model from MLflow (DagsHub)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("citibike_trip_prediction")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mae ASC"], max_results=1)
best_run_id = runs[0].info.run_id
model_uri = f"runs:/{best_run_id}/model"

model = mlflow.pyfunc.load_model(model_uri)

# Step 6: Predict
recent_df["predicted_ride_count"] = model.predict(X_recent)

# Step 7: Save predictions
output_path = "./data/predictions/predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
recent_df[["start_station_id", "datetime", "predicted_ride_count"]].to_csv(output_path, index=False)

print(f"✅ Predictions saved to {output_path}")
