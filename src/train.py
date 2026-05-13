import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from codecarbon import EmissionsTracker


def train_models(
    data_path: str,
    target_column: str = "Production",
    model_output_path: str = "models/best_model.pkl",
    metrics_output_path: str = "reports/metrics.json",
    predictions_output_path: str = "reports/predictions.csv"
):
    """
    Train multiple regression models using time-based split,
    track carbon emissions, save the best pipeline,
    save metrics, and save predictions for dashboard.
    """

    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    if "Date" not in df.columns:
        raise ValueError("Date column not found. Time-based split requires Date column.")

    df["Date"] = pd.to_datetime(df["Date"])

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values(["DateTime", "Source"]).reset_index(drop=True)
    else:
        df = df.sort_values(["Date", "Start_Hour", "Source"]).reset_index(drop=True)

    drop_columns = [target_column, "Date", "DateTime"]

    X = df.drop(columns=[col for col in drop_columns if col in df.columns])
    y = df[target_column]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumerical features:")
    print(numerical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = df["Date"].iloc[split_index:].reset_index(drop=True)

    if "DateTime" in df.columns:
        test_datetimes = df["DateTime"].iloc[split_index:].reset_index(drop=True)
        train_start = df["DateTime"].iloc[:split_index].min()
        train_end = df["DateTime"].iloc[:split_index].max()
        test_start = df["DateTime"].iloc[split_index:].min()
        test_end = df["DateTime"].iloc[split_index:].max()
    else:
        test_datetimes = None
        train_start = df["Date"].iloc[:split_index].min()
        train_end = df["Date"].iloc[:split_index].max()
        test_start = df["Date"].iloc[split_index:].min()
        test_end = df["Date"].iloc[split_index:].max()

    print("\nTime-based split completed.")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Train date range: {train_start} to {train_end}")
    print(f"Test date range: {test_start} to {test_end}")

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    best_pipeline = None
    best_model_name = None
    best_r2 = float("-inf")
    best_predictions = None

    os.makedirs("reports", exist_ok=True)

    tracker = EmissionsTracker(
        project_name="GreenGrid AI - Energy Production Forecasting",
        output_dir="reports",
        output_file="emissions.csv",
        log_level="error"
    )

    tracker.start()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)

        results[model_name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4)
        }

        print(f"{model_name} Results:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipeline
            best_model_name = model_name
            best_predictions = y_pred

    emissions = tracker.stop()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

    joblib.dump(best_pipeline, model_output_path)

    predictions_df = X_test.copy()

    predictions_df["Date"] = test_dates.values

    if test_datetimes is not None:
        predictions_df["DateTime"] = test_datetimes.values

    predictions_df["Actual_Production"] = y_test.values
    predictions_df["Predicted_Production"] = best_predictions
    predictions_df["Error"] = (
        predictions_df["Actual_Production"] - predictions_df["Predicted_Production"]
    )
    predictions_df["Absolute_Error"] = predictions_df["Error"].abs()

    predictions_df.to_csv(predictions_output_path, index=False)

    final_report = {
        "best_model": best_model_name,
        "best_r2_score": round(best_r2, 4),
        "estimated_emissions_kg_co2": round(emissions, 6),
        "all_model_results": results
    }

    with open(metrics_output_path, "w") as file:
        json.dump(final_report, file, indent=4)

    print("\nTraining completed.")
    print(f"Best model: {best_model_name}")
    print(f"Best R2 Score: {best_r2:.4f}")
    print(f"Estimated CO2 emissions: {emissions:.6f} kg")
    print(f"Model saved at: {model_output_path}")
    print(f"Metrics saved at: {metrics_output_path}")
    print(f"Predictions saved at: {predictions_output_path}")

    return final_report