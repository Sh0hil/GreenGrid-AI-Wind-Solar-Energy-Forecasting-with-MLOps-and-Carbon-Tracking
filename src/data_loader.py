import os
import pandas as pd

def load_data(file_path)-> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File is not found at path: {file_path}")

    if not file_path.endswith('.csv'):
        raise ValueError(f"File must be a CSV. Provided file: {file_path}")

    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError(f"The file at {file_path} is empty.")

    print("Data loaded successfully.")
    print(f"Data shape: {df.shape}")
    print(f"Data Columns: ")
    print(df.columns.tolist())

    return df

def inspect_data(df: pd.DataFrame) -> None:
    """Inspect the loaded data for basic information.
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """

    print("\n First 5 rows of the data:")
    print(df.head())    
    
    print("\n Dataset info:")
    print(df.info())

    print("\n Missing values)")
    print(df.isnull().sum())

    print("\n Duplicate rows:")
    print(df.duplicated().sum())

    print("\n Statistical summary:")
    print(df.describe(include='all'))