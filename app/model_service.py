import joblib
import pandas as pd


MODEL_PATH = "models/best_model.pkl"


class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        prediction = self.model.predict(df)[0]
        return round(float(prediction), 2)