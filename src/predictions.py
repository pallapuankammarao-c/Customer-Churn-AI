"""
predictions.py
==============
Inference engine for customer churn risk prediction, probability estimation,
risk tier categorization, and actionable business retention recommendations.
"""

import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_trained_model, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# Risk classification thresholds (configurable business policy)
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60


def categorize_risk(probability: float) -> Tuple[str, str, str]:
    """
    Categorizes churn probability into risk tiers.
    Thresholds:
      - < 0.30 : Low Risk (Green)
      - 0.30 - 0.60 : Medium Risk (Amber)
      - > 0.60 : High Risk (Red)

    Note: These thresholds are configurable business guidelines,
    not universal mathematical constants.
    """
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk", "#10B981", "normal"
    elif probability <= HIGH_RISK_THRESHOLD:
        return "Medium Risk", "#F59E0B", "attention"
    else:
        return "High Risk", "#EF4444", "urgent"


def get_retention_recommendations(customer_data: Dict[str, Any], probability: float) -> Dict[str, Any]:
    """
    Formulates targeted, data-backed retention suggestions based on customer attributes
    and churn probability.
    Note: These are strategic business proposals based on historical correlations,
    not guaranteed causal outcomes.
    """
    actions = []
    urgency = "Standard Monitoring"

    # Contract incentive
    if customer_data.get("Contract") == "Month-to-month":
        actions.append(
            "Contract Incentive: Offer a discounted 1-year or 2-year commitment loyalty plan."
        )

    # Billing friction
    if customer_data.get("PaymentMethod") == "Electronic check":
        actions.append(
            "Payment Optimization: Encourage automated recurring payment (ACH/Credit Card) with a one-time bill credit."
        )

    # High charge review
    monthly_charge = float(customer_data.get("MonthlyCharges", 0))
    if monthly_charge > 75.0:
        actions.append(
            "Value Proposition Audit: Review high monthly charges ($>75) and offer tailored bundle or family plan options."
        )

    # Onboarding / Tenure support
    tenure = int(customer_data.get("tenure", 0))
    if tenure <= 12:
        actions.append(
            "Early Lifecycle Onboarding: Assign priority onboarding support and schedule proactive check-ins during the first 90 days."
        )

    # Tech support & security adoption
    if customer_data.get("TechSupport") == "No":
        actions.append(
            "Support Engagement: Offer complimentary 60-day TechSupport & DeviceProtection trial to deepen service integration."
        )

    if probability > HIGH_RISK_THRESHOLD:
        urgency = "Urgent Intervention Required"
        if not actions:
            actions.append("Direct Account Manager Outreach: Initiate proactive customer satisfaction survey.")
    elif probability >= LOW_RISK_THRESHOLD:
        urgency = "Proactive Retention Recommended"
        if not actions:
            actions.append("Engagement Newsletter: Highlight unused benefits and product updates.")
    else:
        urgency = "Maintain Good Relationship"
        actions = [
            "Customer Delight: Thank customer for loyalty; offer loyalty perks or referral bonus.",
            "Upsell Evaluation: Candidate for premium speed or add-on product trials when appropriate.",
        ]

    return {
        "urgency": urgency,
        "actions": actions,
    }


def predict_single_customer(
    customer_data: Dict[str, Any],
    model_payload: Any = None,
    df: Any = None
) -> Dict[str, Any]:
    """
    Runs model inference for a single customer record and returns probability,
    classification, risk tier, and tailored recommendations.
    Accepts pipeline directly, payload dict, or loads saved model if None.
    """
    if model_payload is None:
        model_payload = load_trained_model()

    if isinstance(model_payload, tuple):
        pipeline = model_payload[0]
    elif hasattr(model_payload, "predict_proba"):
        pipeline = model_payload
    elif isinstance(model_payload, dict) and "pipeline" in model_payload:
        pipeline = model_payload["pipeline"]
    else:
        raise ValueError(f"Unrecognized model payload type: {type(model_payload)}")

    input_df = pd.DataFrame([customer_data])

    # Ensure all required features are present
    for col in NUMERICAL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0.0
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = "No"
        input_df[col] = input_df[col].astype(str)

    # Model inference
    churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
    churn_pred = int(churn_prob >= 0.50)
    prediction_label = "Likely to Churn" if churn_pred == 1 else "Likely to Stay"

    risk_tier, risk_color, risk_urgency = categorize_risk(churn_prob)
    recommendations = get_retention_recommendations(customer_data, churn_prob)

    # Identify contributing factors (heuristics based on known model drivers)
    contributing_factors = []
    if customer_data.get("Contract") == "Month-to-month":
        contributing_factors.append("Month-to-month contract (historically elevated churn)")
    if customer_data.get("PaymentMethod") == "Electronic check":
        contributing_factors.append("Electronic check billing method (high churn segment)")
    if float(customer_data.get("tenure", 0)) < 12:
        contributing_factors.append("Low customer tenure (< 12 months, highest risk window)")
    if float(customer_data.get("MonthlyCharges", 0)) > 75:
        contributing_factors.append("Above-average monthly charges (price sensitivity factor)")
    if customer_data.get("InternetService") == "Fiber optic":
        contributing_factors.append("Fiber optic internet tier (high churn rate category)")
    if customer_data.get("OnlineSecurity") == "No":
        contributing_factors.append("Absence of Online Security add-on service")

    return {
        "churn_probability": churn_prob,
        "churn_probability_pct": round(churn_prob * 100, 1),
        "prediction": prediction_label,
        "risk_tier": risk_tier,
        "risk_category": risk_tier,
        "risk_color": risk_color,
        "urgency": recommendations["urgency"],
        "recommendations": recommendations["actions"],
        "contributing_factors": contributing_factors,
    }


def batch_predict_customers(df: pd.DataFrame, model_payload: Any = None) -> pd.DataFrame:
    """
    Computes churn probability, prediction, and risk tier for an entire dataset.
    Returns enhanced DataFrame suitable for export and aggregate analysis.
    """
    if model_payload is None:
        model_payload = load_trained_model()

    if isinstance(model_payload, tuple):
        pipeline = model_payload[0]
    elif hasattr(model_payload, "predict_proba"):
        pipeline = model_payload
    elif isinstance(model_payload, dict) and "pipeline" in model_payload:
        pipeline = model_payload["pipeline"]
    else:
        raise ValueError(f"Unrecognized model payload type: {type(model_payload)}")

    inference_df = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()

    # Predict probabilities
    probs = pipeline.predict_proba(inference_df)[:, 1]
    output_df = df.copy()
    output_df["Churn_Probability"] = (probs * 100).round(1)
    output_df["Predicted_Churn"] = np.where(probs >= 0.50, "Yes", "No")

    # Risk tiers
    conditions = [
        probs < LOW_RISK_THRESHOLD,
        (probs >= LOW_RISK_THRESHOLD) & (probs <= HIGH_RISK_THRESHOLD),
        probs > HIGH_RISK_THRESHOLD,
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk"]
    output_df["Risk_Level"] = np.select(conditions, choices, default="Medium Risk")
    output_df["Risk_Tier"] = output_df["Risk_Level"]

    return output_df


# Function alias expected by app.py
score_entire_dataset = batch_predict_customers


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 256.50,
    }
    result = predict_single_customer(sample_customer)
    print("Single Prediction Result:")
    for k, v in result.items():
        print(f"{k}: {v}")
