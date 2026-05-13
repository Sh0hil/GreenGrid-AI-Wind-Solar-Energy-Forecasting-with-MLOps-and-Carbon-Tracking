import os
import requests
import numpy as np
import json
import pandas as pd
import streamlit as st
import plotly.express as px


# -------------------------------------------------------------------
# Backend API Configuration
# -------------------------------------------------------------------
# Local development:
#   API_URL=http://127.0.0.1:8000
#
# Docker / AWS EC2 deployment:
#   BACKEND_URL=http://green-grid-ai-backend:8000
#
# This supports both names so the app works locally and inside Docker.
API_URL = (
    os.getenv("BACKEND_URL")
    or os.getenv("API_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")


st.set_page_config(
    page_title="GreenGrid AI Dashboard",
    page_icon="⚡",
    layout="wide"
)


@st.cache_data
def load_metrics():
    with open("reports/metrics.json", "r") as file:
        return json.load(file)


@st.cache_data
def load_predictions():
    return pd.read_csv("reports/predictions.csv")


@st.cache_data
def load_feature_importance():
    return pd.read_csv("reports/feature_importance.csv")


@st.cache_data
def load_emissions():
    try:
        return pd.read_csv("reports/emissions.csv")
    except FileNotFoundError:
        return None


def check_backend_health() -> bool:
    """
    Check whether the FastAPI backend is reachable.
    It first tries /health. If /health is not available, it tries /docs.
    """
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            return True

        docs_response = requests.get(f"{API_URL}/docs", timeout=5)
        return docs_response.status_code == 200

    except requests.exceptions.RequestException:
        return False


metrics = load_metrics()
predictions = load_predictions()
feature_importance = load_feature_importance()
emissions = load_emissions()


st.title("GreenGrid AI")
st.subheader("Wind & Solar Energy Production Forecasting with MLOps and Carbon Tracking")

st.markdown(
    """
    This dashboard monitors model performance, renewable energy production forecasts,
    feature importance, prediction errors, and estimated carbon emissions from training.
    """
)


with st.sidebar:
    st.header("Backend Status")
    st.caption("Current backend API URL:")
    st.code(API_URL)

    if check_backend_health():
        st.success("Backend connected")
    else:
        st.error("Backend not reachable")

best_model = metrics["best_model"]
best_r2 = metrics["best_r2_score"]
carbon_emissions = metrics["estimated_emissions_kg_co2"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Model", best_model)

with col2:
    st.metric("Best R² Score", best_r2)

with col3:
    st.metric("Estimated CO₂ Emissions", f"{carbon_emissions} kg")


st.divider()

st.header("Model Comparison")

model_results = metrics["all_model_results"]
model_df = pd.DataFrame(model_results).T.reset_index()
model_df = model_df.rename(columns={"index": "Model"})

st.dataframe(model_df, use_container_width=True)

fig_model = px.bar(
    model_df,
    x="Model",
    y="R2_Score",
    title="Model R² Score Comparison",
    text="R2_Score"
)

st.plotly_chart(fig_model, use_container_width=True)


st.divider()

st.header("Actual vs Predicted Production")

predictions["DateTime"] = pd.to_datetime(predictions["DateTime"])

source_options = predictions["Source"].unique().tolist()
selected_source = st.selectbox("Select Energy Source", source_options)

filtered_predictions = predictions[predictions["Source"] == selected_source].copy()

fig_forecast = px.line(
    filtered_predictions.head(500),
    x="DateTime",
    y=["Actual_Production", "Predicted_Production"],
    title=f"Actual vs Predicted Production - {selected_source}"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.caption("Showing first 500 test records for smoother visualization.")


st.divider()

st.header("Live Energy Production Prediction")

st.markdown(
    """
    Select the energy source, date, hour, and recent production history.
    The dashboard will automatically generate time-based features and send them
    to the FastAPI model endpoint for live prediction.
    """
)


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


col1, col2, col3 = st.columns(3)

with col1:
    source = st.selectbox("Energy Source", ["Solar", "Wind"])

with col2:
    selected_date = st.date_input("Select Date")

with col3:
    start_hour = st.slider("Start Hour", 0, 23, 10)


selected_date = pd.to_datetime(selected_date)

end_hour = start_hour + 1 if start_hour < 23 else 0

year = selected_date.year
month = selected_date.month
day = selected_date.day

day_of_year = selected_date.dayofyear
week_of_year = int(selected_date.isocalendar().week)

day_name = selected_date.day_name()
month_name = selected_date.month_name()
season = get_season(month)

is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0

hour_sin = np.sin(2 * np.pi * start_hour / 24)
hour_cos = np.cos(2 * np.pi * start_hour / 24)

day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)


st.subheader("Auto-Generated Time Features")

feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

with feature_col1:
    st.metric("Day of Year", day_of_year)
    st.metric("Week of Year", week_of_year)

with feature_col2:
    st.metric("Day Name", day_name)
    st.metric("Month Name", month_name)

with feature_col3:
    st.metric("Season", season)
    st.metric("Weekend", "Yes" if is_weekend == 1 else "No")

with feature_col4:
    st.metric("End Hour", end_hour)
    st.metric("Year", year)


st.subheader("Recent Production History")

hist_col1, hist_col2, hist_col3 = st.columns(3)

with hist_col1:
    production_lag_1 = st.number_input(
        "Previous Hour Production",
        value=5200.0,
        help="Energy production recorded in the previous hour."
    )

    production_lag_2 = st.number_input(
        "Previous 2 Hours Production",
        value=5000.0,
        help="Energy production recorded two hours ago."
    )

    production_lag_24 = st.number_input(
        "Same Hour Yesterday",
        value=4800.0,
        help="Energy production at the same hour on the previous day."
    )

with hist_col2:
    production_lag_168 = st.number_input(
        "Same Hour Last Week",
        value=5100.0,
        help="Energy production at the same hour one week ago."
    )

    production_rolling_mean_3 = st.number_input(
        "3-Hour Rolling Mean",
        value=5050.0,
        help="Average production over the previous 3 hours."
    )

with hist_col3:
    production_rolling_mean_24 = st.number_input(
        "24-Hour Rolling Mean",
        value=4900.0,
        help="Average production over the previous 24 hours."
    )

    production_rolling_mean_168 = st.number_input(
        "Weekly Rolling Mean",
        value=5000.0,
        help="Average production over the previous 168 hours, which is one week."
    )


input_payload = {
    "Start_Hour": start_hour,
    "End_Hour": end_hour,
    "Source": source,
    "Day_of_Year": day_of_year,
    "Day_Name": day_name,
    "Month_Name": month_name,
    "Season": season,
    "Year": year,
    "Month": month,
    "Day": day,
    "Week_of_Year": week_of_year,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_of_year_sin": day_of_year_sin,
    "day_of_year_cos": day_of_year_cos,
    "is_weekend": is_weekend,
    "production_lag_1": production_lag_1,
    "production_lag_2": production_lag_2,
    "production_lag_24": production_lag_24,
    "production_lag_168": production_lag_168,
    "production_rolling_mean_3": production_rolling_mean_3,
    "production_rolling_mean_24": production_rolling_mean_24,
    "production_rolling_mean_168": production_rolling_mean_168
}


with st.expander("View Model Input Features"):
    st.json(input_payload)


if st.button("Predict Energy Production"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=input_payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()

            st.success(
                f"Predicted {source} Production: {result['predicted_production']}"
            )

            st.metric(
                label="Predicted Production",
                value=result["predicted_production"]
            )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            f"FastAPI backend is not reachable at {API_URL}. "
            "If running locally, start it using: python -m uvicorn app.main:app --reload"
        )

    except requests.exceptions.Timeout:
        st.error(
            f"Request timed out while connecting to FastAPI backend at {API_URL}."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Unexpected API request error: {error}")