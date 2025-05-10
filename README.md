# 🚲 Citi Bike Trip Prediction System

A fully automated MLOps project that predicts hourly Citi Bike ride demand at top NYC stations using public trip history data. Built end-to-end with Hopsworks, MLflow, LightGBM, GitHub Actions, and Streamlit.

> **Final Semester Project – University at Buffalo (Due: March 10, 2025)**
> **Review Call with TA: May 10–11, 2025**

---

## 🔗 Live Demo Apps

* **📈 Prediction Dashboard**
  [View App →](https://citibike-predictor-th4yfqejvva5fayvqvptyn.streamlit.app/)
  Select a station to view hourly ride predictions (last 48 hours).

* **📊 Model Monitoring Dashboard**
  [View App →](https://citibike-predictor-hyfy8exskqbkxxvuda798c.streamlit.app/)
  Track model MAE across experiments and visualize ride forecasts.

---

## 📦 Project Structure

```
├── .github/workflows/           # GitHub Actions for automation
│   ├── feature-engineering.yml
│   ├── train-model.yml
│   └── batch-inference.yml
├── data/                        # Raw + processed data (locally or from Hopsworks)
├── notebooks/                   # EDA and dev notebooks
├── src/                         # Core pipeline scripts
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── inference.py
│   └── utils.py
├── streamlit_app/              # Streamlit dashboards
│   ├── prediction_app.py
│   └── monitoring_app.py
├── .env                        # Secrets (use dotenv)
├── requirements.txt
├── README.md
└── config.toml                 # Streamlit UI config
```

---

## 📊 Features & Pipelines

### 🛍️ Data Engineering (Hopsworks)

* Fetched raw trip data from [Citi Bike NYC](https://citibikenyc.com/system-data)
* Filtered top 3 stations with highest ride volume
* Cleaned, aggregated by hour, and engineered features (lags, rolling stats)
* Stored feature sets in Hopsworks Feature Store

### 🫠 Modeling & MLflow

* **Baseline model**: 7-day mean
* **Lag model**: LightGBM with 28 lag features
* **Reduced model**: LightGBM with top 10 features (feature importance)
* Tracked experiments & metrics in MLflow (hosted via DagsHub)

### ♻️ Automation & CI/CD

* Scheduled **GitHub Actions** for:

  * Daily feature engineering
  * Weekly model retraining
  * Hourly inference

### 📈 Frontend & Monitoring

* **Prediction App**: Interactive dashboard for station-wise forecast
* **Monitoring App**: Tracks model performance (MAE) and prediction logs
* Deployed on Streamlit Cloud

---

## 🛠️ Technologies Used

| Layer            | Tools/Services               |
| ---------------- | ---------------------------- |
| Data Engineering | Python, Pandas, Hopsworks FS |
| Modeling         | LightGBM, MLflow (DagsHub)   |
| Automation       | GitHub Actions               |
| Deployment       | Streamlit Cloud              |
| Tracking         | MLflow, Hopsworks FS         |

---

## 📌 How to Run Locally

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/citibike-predictor.git
   cd citibike-predictor
   ```

2. Create and activate Conda env:

   ```bash
   conda create --name bikepred python=3.11
   conda activate bikepred
   pip install -r requirements.txt
   ```

3. Add your `.env` file with Hopsworks and MLflow credentials.

4. Run Streamlit apps locally:

   ```bash
   streamlit run streamlit_app/prediction_app.py
   # OR
   streamlit run streamlit_app/monitoring_app.py
   ```

---

## 🤛 Final Notes

* ✅ Complete MLOps pipeline from data ingestion to live dashboard
* 🗓️ Final TA review scheduled for May 10–11, 2025
* 🌟 Exceptional performance may boost final semester grade

---

## 📧 Contact

**Atul Sharma**
[LinkedIn](https://www.linkedin.com/in/atulsharma/) | atul.ammy01 \[at] gmail.com
University at Buffalo – MS Data Science
