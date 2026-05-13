# GreenGrid AI: Wind & Solar Energy Production Forecasting with MLOps

GreenGrid AI is an end-to-end machine learning and MLOps project that predicts wind and solar energy production using historical production patterns, time-based features, lag features, and rolling averages.

The project includes model training, carbon emission tracking, FastAPI model serving, Streamlit dashboard, Docker containerization, and model performance monitoring.

---

## Project Overview

Renewable energy production depends on source type, time, seasonality, and recent production behavior. This project predicts renewable energy production using historical wind and solar production data from 2020 to 2025.

The model uses a time-based train-test split, where older data is used for training and future data is used for testing.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Streamlit
- Plotly
- CodeCarbon
- Docker
- Docker Compose

---

## Dataset

Dataset used: Energy Production Dataset

Main columns:

- Date
- Start_Hour
- End_Hour
- Source
- Day_of_Year
- Day_Name
- Month_Name
- Season
- Production

Target column:

```text
Production
```

---

## Feature Engineering

The project creates:

- Date features: year, month, day, week of year
- Cyclical time features: hour_sin, hour_cos, day_of_year_sin, day_of_year_cos
- Lag features:
  - production_lag_1
  - production_lag_2
  - production_lag_24
  - production_lag_168
- Rolling features:
  - production_rolling_mean_3
  - production_rolling_mean_24
  - production_rolling_mean_168

---

## Models Trained

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Results

Best model:

```text
XGBoost
```

Performance:

```text
R² Score: 0.9561
MAE: 510.60
RMSE: 916.38
Estimated CO₂ emissions: 0.000266 kg
```

---

## Feature Importance

Top features:

```text
production_lag_1
production_rolling_mean_3
production_lag_2
Source_Wind
Source_Solar
hour_cos
Start_Hour
```

This shows that recent production history is the most important factor in forecasting renewable energy production.

---

## Project Structure

```text
green-grid-ai/
│
├── app/
│   ├── main.py
│   ├── schema.py
│   └── model_service.py
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── model_analysis.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── metrics.json
│   ├── emissions.csv
│   ├── predictions.csv
│   └── feature_importance.csv
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Run Locally

### Train the model

```bash
python main.py
python train_pipeline.py
python analyze_model.py
```

### Run FastAPI backend

```bash
python -m uvicorn app.main:app --reload
```

FastAPI docs:

```text
http://localhost:8000/docs
```

### Run Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

---

## Run With Docker

Make sure Docker Desktop is running.

```bash
docker compose up --build
```

Open:

```text
FastAPI: http://localhost:8000/docs
Streamlit: http://localhost:8501
```

---

## API Endpoints

### Health Check

```text
GET /health
```

### Model Info

```text
GET /model-info
```

### Prediction

```text
POST /predict
```

Sample input:

```json
{
  "Start_Hour": 10,
  "End_Hour": 11,
  "Source": "Solar",
  "Day_of_Year": 150,
  "Day_Name": "Friday",
  "Month_Name": "May",
  "Season": "Spring",
  "Year": 2025,
  "Month": 5,
  "Day": 30,
  "Week_of_Year": 22,
  "hour_sin": 0.5,
  "hour_cos": -0.866,
  "day_of_year_sin": 0.53,
  "day_of_year_cos": -0.84,
  "is_weekend": 0,
  "production_lag_1": 5200,
  "production_lag_2": 5000,
  "production_lag_24": 4800,
  "production_lag_168": 5100,
  "production_rolling_mean_3": 5050,
  "production_rolling_mean_24": 4900,
  "production_rolling_mean_168": 5000
}
```

---

## Dashboard Features

The Streamlit dashboard includes:

- Model comparison
- Best model metrics
- Actual vs predicted production graph
- Live prediction form
- Auto-generated time features
- Feature importance
- Carbon emission tracking

---

## Green AI Component

This project uses CodeCarbon to estimate CO₂ emissions generated during model training. This helps evaluate the model not only by accuracy, but also by computational sustainability.

---

## Future Improvements

- Add MLflow or Weights & Biases experiment tracking
- Deploy on AWS EC2
- Add CI/CD using GitHub Actions
- Add model retraining pipeline
- Add database support for storing predictions
- Add separate models for solar and wind production

---

## Resume Bullet

Built GreenGrid AI, an end-to-end MLOps project for wind and solar energy production forecasting using XGBoost, FastAPI, Streamlit, Docker, and CodeCarbon. Achieved an R² score of 0.9561 using time-based train-test split, lag features, and rolling averages while tracking model training CO₂ emissions.
