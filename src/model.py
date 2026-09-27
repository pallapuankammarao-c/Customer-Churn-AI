"""
model.py
========
Machine learning training and evaluation pipeline for customer churn.
Strictly avoids data leakage by encapsulating all scaling and encoding
inside Scikit-learn Pipeline and ColumnTransformer on stratified splits.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

try:
    from src.data_cleaning import load_raw_data, clean_churn_data
except ImportError:
    from data_cleaning import load_raw_data, clean_churn_data


# Feature definitions
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
TARGET_COLUMN = "Churn"

# Aliases expected by app.py
NUMERICAL_COLS = NUMERICAL_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES


class ModelArtifact(dict):
    """Container for model pipeline and telemetry that can act as both dict and 2-tuple (pipeline, payload)."""
    def __iter__(self):
        # Allows `model, metrics = load_trained_model(...)`
        return iter((self["pipeline"], self))


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to apply StandardScaler to numerical features
    and OneHotEncoder to categorical features without data leakage.
    """
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    cat_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    df: pd.DataFrame,
    model_save_path: str = "models/churn_model.pkl",
) -> Dict[str, Any]:
    """
    Splits data (80/20 stratified), trains Logistic Regression and Random Forest,
    evaluates both on unseen test data, selects the best model, and saves it.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = (df[TARGET_COLUMN].str.lower() == "yes").astype(int)

    # Stratified 80% train / 20% test split to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=10, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    fitted_pipelines = {}
    comparison_records = []

    for name, clf in models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)

        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "confusion_matrix": cm.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "y_prob": y_prob.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

        comparison_records.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
            }
        )
        fitted_pipelines[name] = pipeline

    comparison_df = pd.DataFrame(comparison_records)

    # Select best model based on ROC-AUC and balanced F1 score
    # Random Forest vs Logistic Regression comparison
    best_model_name = comparison_df.sort_values(by=["ROC-AUC", "F1 Score"], ascending=False).iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    # Extract feature names & importances/weights for explainability
    preprocessor = best_pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        raw_importance = classifier.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": all_feature_names, "Importance": raw_importance}
        ).sort_values(by="Importance", ascending=False)
    elif hasattr(classifier, "coef_"):
        raw_coef = classifier.coef_[0]
        importance_df = pd.DataFrame(
            {"Feature": all_feature_names, "Importance": np.abs(raw_coef), "Coefficient": raw_coef}
        ).sort_values(by="Importance", ascending=False)
    else:
        importance_df = pd.DataFrame({"Feature": all_feature_names, "Importance": 0.0})

    save_payload = {
        "best_model_name": best_model_name,
        "pipeline": best_pipeline,
        "models": results,
        "all_results": results,
        "comparison_table": comparison_df.to_dict(orient="records"),
        "train_size": len(X_train),
        "train_samples": len(X_train),
        "test_size": len(X_test),
        "test_samples": len(X_test),
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names": all_feature_names,
        "top_features": importance_df.head(15).to_dict(orient="records"),
    }

    joblib.dump(save_payload, model_save_path)

    return ModelArtifact(save_payload)


def load_trained_model(
    df_or_path=None, model_path: str = "models/churn_model.pkl"
) -> ModelArtifact:
    """
    Loads saved model artifact dictionary. If absent, runs training automatically.
    Returns ModelArtifact which supports both dict indexing (artifact['pipeline'])
    and tuple unpacking (pipeline, metrics = load_trained_model()).
    """
    actual_path = model_path
    df = None
    if isinstance(df_or_path, str):
        actual_path = df_or_path
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path

    if not os.path.exists(actual_path):
        if df is None:
            raw_df = load_raw_data()
            df, _ = clean_churn_data(raw_df)
        payload = train_and_evaluate_models(df, actual_path)
    else:
        payload = joblib.load(actual_path)

    # Ensure required compatibility keys
    payload.setdefault("models", payload.get("all_results", {}))
    payload.setdefault("all_results", payload.get("models", {}))
    payload.setdefault("train_samples", payload.get("train_size", 0))
    payload.setdefault("test_samples", payload.get("test_size", 0))

    return ModelArtifact(payload)


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df, _ = clean_churn_data(raw_df)
    payload = train_and_evaluate_models(clean_df)
    print("Model Training & Evaluation Completed.")
    print("Comparison Table:")
    print(pd.DataFrame(payload["comparison_table"]))
    print(f"Selected Best Model: {payload['best_model_name']}")
