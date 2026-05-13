from fastapi import FastAPI
from app.schema import EnergyPredictionInput
from app.model_service import ModelService


app = FastAPI(
    title="GreenGrid AI API",
    description="Wind & Solar Energy Production Forecasting API",
    version="1.0.0"
)

model_service = ModelService()


@app.get("/")
def home():
    return {
        "message": "Welcome to GreenGrid AI API",
        "project": "Wind & Solar Energy Production Forecasting with MLOps"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "target": "Production",
        "problem_type": "Regression",
        "features": 23
    }


@app.post("/predict")
def predict_energy(input_data: EnergyPredictionInput):
    input_dict = input_data.model_dump()
    prediction = model_service.predict(input_dict)

    return {
        "predicted_production": prediction,
        "unit": "energy production units",
        "source": input_data.Source
    }