import pandas as pd
import yaml


def read_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = read_config()
    file_path = config["data"]["raw_data_path"]

    df = pd.read_csv(file_path)

    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()