from pydantic import BaseModel


class EnergyPredictionInput(BaseModel):
    Start_Hour: int
    End_Hour: int
    Source: str
    Day_of_Year: int
    Day_Name: str
    Month_Name: str
    Season: str
    Year: int
    Month: int
    Day: int
    Week_of_Year: int
    hour_sin: float
    hour_cos: float
    day_of_year_sin: float
    day_of_year_cos: float
    is_weekend: int
    production_lag_1: float
    production_lag_2: float
    production_lag_24: float
    production_lag_168: float
    production_rolling_mean_3: float
    production_rolling_mean_24: float
    production_rolling_mean_168: float