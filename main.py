import os
import yaml

from src.data_loader import load_data, inspect_data
from src.preprocessing import preprocess_data, split_features_target
from src.feature_engineering import create_features


def read_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def main():
    config = read_config()

    raw_data_path = config["data"]["raw_data_path"]
    processed_data_path = config["data"]["processed_data_path"]

    df = load_data(raw_data_path)

    inspect_data(df)

    clean_df = preprocess_data(df)

    feature_df = create_features(clean_df)

    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    feature_df.to_csv(processed_data_path, index=False)

    print(f"\nProcessed data saved at: {processed_data_path}")

    X, y = split_features_target(feature_df)

    print("\nFinal feature columns:")
    print(X.columns.tolist())

    print("\nTarget column:")
    print("Production")


if __name__ == "__main__":
    main()