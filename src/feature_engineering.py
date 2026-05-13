import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based, cyclical, lag, and rolling features
    for energy production forecasting.
    """

    df = df.copy()

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Create DateTime from Date + Start_Hour
    df["DateTime"] = df["Date"] + pd.to_timedelta(df["Start_Hour"], unit="h")

    # Sort properly for time-series features
    df = df.sort_values(["Source", "DateTime"]).reset_index(drop=True)

    # Basic date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["Start_Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Start_Hour"] / 24)

    # Cyclical day of year encoding
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365)

    # Weekend feature
    df["is_weekend"] = df["Day_Name"].isin(["Saturday", "Sunday"]).astype(int)

    # Lag features source-wise
    df["production_lag_1"] = df.groupby("Source")["Production"].shift(1)
    df["production_lag_2"] = df.groupby("Source")["Production"].shift(2)
    df["production_lag_24"] = df.groupby("Source")["Production"].shift(24)
    df["production_lag_168"] = df.groupby("Source")["Production"].shift(168)

    # Rolling features source-wise
    df["production_rolling_mean_3"] = (
        df.groupby("Source")["Production"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["production_rolling_mean_24"] = (
        df.groupby("Source")["Production"]
        .shift(1)
        .rolling(window=24)
        .mean()
    )

    df["production_rolling_mean_168"] = (
        df.groupby("Source")["Production"]
        .shift(1)
        .rolling(window=168)
        .mean()
    )

    # Remove rows with NaN created by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    print("Feature engineering completed.")
    print(f"Shape after feature engineering: {df.shape}")

    return df