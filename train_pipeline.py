import yaml
from src.train import train_models


def read_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    config = read_config()

    processed_data_path = config["data"]["processed_data_path"]

    train_models(
        data_path=processed_data_path,
        target_column="Production",
        model_output_path="models/best_model.pkl",
        metrics_output_path="reports/metrics.json"
    )


if __name__ == "__main__":
    main()