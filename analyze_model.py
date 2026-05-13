from src.model_analysis import save_feature_importance


def main():
    save_feature_importance(
        model_path="models/best_model.pkl",
        output_path="reports/feature_importance.csv"
    )


if __name__ == "__main__":
    main()