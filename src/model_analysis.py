import os
import json
import joblib
import pandas as pd


def save_feature_importance(
    model_path: str = "models/best_model.pkl",
    output_path: str = "reports/feature_importance.csv"
):
    """
    Extract and save feature importance from the best trained model pipeline.
    Works for tree-based models like Random Forest, Gradient Boosting, and XGBoost.
    """

    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print("This model does not support feature importance.")
        return

    cat_features = preprocessor.transformers_[0][2]
    num_features = preprocessor.transformers_[1][2]

    encoder = preprocessor.named_transformers_["cat"]

    encoded_cat_features = encoder.get_feature_names_out(cat_features)

    all_features = list(encoded_cat_features) + list(num_features)

    importance_df = pd.DataFrame({
        "feature": all_features,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    print(f"Feature importance saved at: {output_path}")
    print("\nTop 15 important features:")
    print(importance_df.head(15))