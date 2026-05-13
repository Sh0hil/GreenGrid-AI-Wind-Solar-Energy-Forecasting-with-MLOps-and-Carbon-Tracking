import os
import pandas as pd


def analyze_target_relationships(
    data_path: str,
    target_column: str = "grid_load_demand",
    report_path: str = "reports/correlation_report.csv"
):
    """
    Analyze relationship between features and target column.

    Args:
        data_path (str): Path to processed dataset.
        target_column (str): Target column name.
        report_path (str): Path to save correlation report.
    """

    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # Drop timestamp because correlation works only on numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    correlation = numeric_df.corr()[target_column].sort_values(
        ascending=False
    )

    print("\nTarget column:")
    print(target_column)

    print("\nTop positive correlations:")
    print(correlation.head(10))

    print("\nTop negative correlations:")
    print(correlation.tail(10))

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    correlation.to_csv(report_path)

    print(f"\nCorrelation report saved at: {report_path}")