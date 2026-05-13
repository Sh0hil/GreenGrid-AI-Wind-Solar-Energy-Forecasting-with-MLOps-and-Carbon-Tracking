import pandas as pd


TARGET_COLUMN = "Production"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess energy production dataset.

    Steps:
    1. Convert Date column to datetime.
    2. Sort data by Date and Start_Hour.
    3. Remove duplicates.
    4. Validate target column.
    """

    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError("Date column not found in dataset.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "Start_Hour"]).reset_index(drop=True)

    df = df.drop_duplicates()

    print("Data preprocessing completed.")
    print(f"Final shape after preprocessing: {df.shape}")

    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features X and target y.
    """

    target_column = TARGET_COLUMN

    drop_columns = [
        target_column,
        "Date"
    ]

    X = df.drop(columns=[col for col in drop_columns if col in df.columns])
    y = df[target_column]

    print("Feature-target split completed.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y