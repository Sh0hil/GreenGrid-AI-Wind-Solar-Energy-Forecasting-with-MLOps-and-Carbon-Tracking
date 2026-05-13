import yaml
from src.analyze import analyze_target_relationships


def read_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def main():
    config = read_config()

    processed_data_path = config["data"]["processed_data_path"]

    analyze_target_relationships(
        data_path=processed_data_path,
        target_column="target_next_hour_energy",
        report_path="reports/correlation_report.csv"
    )


if __name__ == "__main__":
    main()