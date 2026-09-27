"""
analysis.py
===========
Exploratory Data Analysis (EDA) and Business KPI engine.
Generates publication-quality interactive Plotly visualizations and metrics.
"""

import os
import sys
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Aesthetic Color Palette
COLOR_RETAINED = "#10B981"  # Emerald
COLOR_CHURNED = "#F43F5E"   # Vibrant Rose / Coral
COLOR_ACCENT = "#3B82F6"    # Indigo / Blue
COLOR_BG = "#0F172A"        # Deep Slate
CHURN_MAP = {"No": COLOR_RETAINED, "Yes": COLOR_CHURNED, 0: COLOR_RETAINED, 1: COLOR_CHURNED}

# Global Color Constants expected by app.py
CHURN_COLOR = "#F43F5E"
RETAIN_COLOR = "#10B981"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"


def compute_business_kpis(df: pd.DataFrame, scored_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Computes core business KPIs directly from the dataset and scored predictions.
    """
    total_customers = len(df)
    churn_count = int((df["Churn"].astype(str).str.lower() == "yes").sum()) if total_customers > 0 else 0
    retained_count = total_customers - churn_count
    churn_rate = round((churn_count / total_customers * 100), 2) if total_customers > 0 else 0.0
    avg_monthly_charges = round(float(df["MonthlyCharges"].mean()), 2) if total_customers > 0 else 0.0
    avg_tenure = round(float(df["tenure"].mean()), 1) if total_customers > 0 else 0.0
    total_mrr = round(float(df["MonthlyCharges"].sum()), 2) if total_customers > 0 else 0.0
    
    churn_mask = df["Churn"].astype(str).str.lower() == "yes"
    mrr_at_risk = round(float(df[churn_mask]["MonthlyCharges"].sum()), 2) if total_customers > 0 else 0.0
    churn_monthly_revenue_loss = mrr_at_risk

    # Compute high-risk customer count & percentage
    high_risk_count = 0
    if scored_df is not None:
        if "Risk_Level" in scored_df.columns:
            high_risk_count = int((scored_df["Risk_Level"] == "High Risk").sum())
        elif "Risk_Tier" in scored_df.columns:
            high_risk_count = int((scored_df["Risk_Tier"] == "High Risk").sum())
        elif "Churn_Probability" in scored_df.columns:
            high_risk_count = int((scored_df["Churn_Probability"] >= 60.0).sum())
    elif "Risk_Level" in df.columns:
        high_risk_count = int((df["Risk_Level"] == "High Risk").sum())
    elif "Risk_Tier" in df.columns:
        high_risk_count = int((df["Risk_Tier"] == "High Risk").sum())
    else:
        # Proxy: Month-to-month and Electronic Check and tenure <= 12
        high_risk_mask = (
            (df["Contract"] == "Month-to-month")
            & (df["PaymentMethod"] == "Electronic check")
            & (df["tenure"] <= 12)
        )
        high_risk_count = int(high_risk_mask.sum())

    high_risk_pct = round((high_risk_count / total_customers * 100), 1) if total_customers > 0 else 0.0

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly_charges,
        "avg_tenure": avg_tenure,
        "total_mrr": total_mrr,
        "mrr_at_risk": mrr_at_risk,
        "churn_monthly_revenue_loss": churn_monthly_revenue_loss,
        "high_risk_count": high_risk_count,
        "high_risk_customers": high_risk_count,
        "high_risk_pct": high_risk_pct,
    }


def calculate_business_kpis(df: pd.DataFrame, high_risk_count: int = None) -> Dict[str, Any]:
    """Backward compatibility alias for compute_business_kpis."""
    return compute_business_kpis(df)


def create_churn_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    1. Overall Churn Distribution Donut Chart.
    """
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Label"] = counts["Status"].map({"No": "Retained", "Yes": "Churned"})

    fig = px.pie(
        counts,
        names="Label",
        values="Count",
        hole=0.55,
        color="Status",
        color_discrete_map={"No": COLOR_RETAINED, "Yes": COLOR_CHURNED},
        title="Overall Customer Retention vs. Churn",
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=50, b=30, l=20, r=20),
    )
    return fig


def create_churn_by_categorical_chart(
    df: pd.DataFrame, category_col: str, title: str, x_label: str
) -> go.Figure:
    """
    Creates normalized 100% stacked bar chart showing churn rates across categories.
    """
    grouped = (
        df.groupby([category_col, "Churn"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["No", "Yes"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["Total"] = grouped["No"] + grouped["Yes"]
    grouped["ChurnRate"] = (grouped["Yes"] / grouped["Total"] * 100).round(1)
    grouped = grouped.sort_values(by="ChurnRate", ascending=False)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped[category_col],
            y=grouped["No"],
            name="Retained",
            marker_color=COLOR_RETAINED,
            hovertemplate="Retained: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped[category_col],
            y=grouped["Yes"],
            name="Churned",
            marker_color=COLOR_CHURNED,
            hovertemplate="Churned: %{y}<br>Churn Rate: %{text}%<extra></extra>",
            text=grouped["ChurnRate"],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title=x_label,
        yaxis_title="Number of Customers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def create_tenure_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    3. Tenure vs Churn Distribution.
    """
    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        barmode="overlay",
        nbins=36,
        color_discrete_map={"No": COLOR_RETAINED, "Yes": COLOR_CHURNED},
        labels={"tenure": "Tenure (Months)", "Churn": "Status"},
        title="Customer Tenure Distribution (Months)",
        opacity=0.75,
    )
    fig.update_layout(
        xaxis_title="Tenure (Months)",
        yaxis_title="Customer Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def create_monthly_charges_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    4. Monthly Charges Distribution by Churn Status.
    """
    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": COLOR_RETAINED, "Yes": COLOR_CHURNED},
        points="outliers",
        labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": "Status"},
        title="Monthly Charges Distribution ($)",
    )
    fig.update_layout(
        xaxis_title="Customer Status",
        yaxis_title="Monthly Charges ($)",
        showlegend=False,
        margin=dict(t=50, b=40, l=20, r=20),
    )
    return fig


def create_scatter_charges_tenure(df: pd.DataFrame) -> go.Figure:
    """
    Tenure vs Monthly Charges with Churn Hue.
    """
    sample_df = df.sample(min(1200, len(df)), random_state=42)
    fig = px.scatter(
        sample_df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": COLOR_RETAINED, "Yes": COLOR_CHURNED},
        opacity=0.6,
        labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"},
        title="Tenure vs. Monthly Charges (Sampled 1,200 Customers)",
    )
    fig.update_layout(
        xaxis_title="Tenure (Months)",
        yaxis_title="Monthly Charges ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=20, r=20),
    )
    return fig


def create_risk_distribution_chart(risk_data) -> go.Figure:
    """
    Customer Risk Category Distribution (Low, Medium, High).
    Accepts either a pandas Series or a DataFrame containing 'Risk_Level' or 'Risk_Tier'.
    """
    if isinstance(risk_data, pd.DataFrame):
        if "Risk_Level" in risk_data.columns:
            risk_series = risk_data["Risk_Level"]
        elif "Risk_Tier" in risk_data.columns:
            risk_series = risk_data["Risk_Tier"]
        elif "risk_category" in risk_data.columns:
            risk_series = risk_data["risk_category"]
        else:
            risk_series = pd.Series(["Low Risk"] * len(risk_data))
    else:
        risk_series = risk_data

    risk_counts = risk_series.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    risk_df = risk_counts.reset_index()
    risk_df.columns = ["RiskLevel", "Count"]

    color_map = {
        "Low Risk": "#10B981",     # Green
        "Medium Risk": "#F59E0B",  # Amber
        "High Risk": "#EF4444",    # Red
    }

    fig = px.bar(
        risk_df,
        x="RiskLevel",
        y="Count",
        color="RiskLevel",
        color_discrete_map=color_map,
        text="Count",
        title="Model-Predicted Customer Risk Distribution",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Risk Tier",
        yaxis_title="Customer Count",
        showlegend=False,
        margin=dict(t=50, b=40, l=20, r=20),
    )
    return fig


# Aliases & dedicated chart builders expected by app.py
create_customer_risk_distribution_chart = create_risk_distribution_chart


def create_churn_by_contract_chart(df: pd.DataFrame) -> go.Figure:
    """Churn Distribution by Contract Type."""
    return create_churn_by_categorical_chart(
        df, "Contract", "Churn Distribution by Contract Type", "Contract Type"
    )


def create_churn_by_tenure_chart(df: pd.DataFrame) -> go.Figure:
    """Tenure vs Churn Distribution."""
    return create_tenure_distribution_chart(df)


def create_churn_by_monthly_charges_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly Charges Distribution by Churn Status."""
    return create_monthly_charges_distribution_chart(df)


def create_churn_by_payment_method_chart(df: pd.DataFrame) -> go.Figure:
    """Churn Distribution by Payment Method."""
    return create_churn_by_categorical_chart(
        df, "PaymentMethod", "Churn Distribution by Payment Method", "Payment Method"
    )


def create_churn_by_internet_service_chart(df: pd.DataFrame) -> go.Figure:
    """Churn Distribution by Internet Service."""
    return create_churn_by_categorical_chart(
        df, "InternetService", "Churn Distribution by Internet Service", "Internet Service"
    )


def create_churn_by_senior_citizen_chart(df: pd.DataFrame) -> go.Figure:
    """Churn Distribution by Senior Citizen Status."""
    return create_churn_by_categorical_chart(
        df, "SeniorCitizen", "Churn Distribution by Senior Citizen Status", "Senior Citizen"
    )


def create_churn_by_tech_support_chart(df: pd.DataFrame) -> go.Figure:
    """Churn Distribution by Tech Support Subscription."""
    return create_churn_by_categorical_chart(
        df, "TechSupport", "Churn Distribution by Tech Support Subscription", "Tech Support"
    )


def create_confusion_matrix_chart(cm_data: list, model_name: str) -> go.Figure:
    """
    Confusion Matrix Heatmap.
    """
    z = cm_data
    x = ["Predicted Retained (0)", "Predicted Churned (1)"]
    y = ["Actual Retained (0)", "Actual Churned (1)"]

    z_text = [[str(val) for val in row] for row in z]

    fig = px.imshow(
        z,
        x=x,
        y=y,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=f"Confusion Matrix: {model_name}",
        text_auto=True,
    )
    fig.update_layout(margin=dict(t=60, b=40, l=20, r=20))
    return fig


def create_roc_curve_chart(all_results: dict) -> go.Figure:
    """
    Comparative ROC Curves.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94A3B8"),
            name="Random Baseline (AUC = 0.50)",
        )
    )

    colors = ["#3B82F6", "#10B981", "#F59E0B"]
    for idx, (m_name, m_data) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        auc_score = m_data.get("roc_auc", 0.0)
        fig.add_trace(
            go.Scatter(
                x=m_data.get("fpr", []),
                y=m_data.get("tpr", []),
                mode="lines",
                name=f"{m_name} (AUC = {auc_score:.3f})",
                line=dict(color=color, width=2.5),
            )
        )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Comparison",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=50, b=80, l=40, r=20),
    )
    return fig


def create_feature_importance_chart(top_features: list) -> go.Figure:
    """
    Top Feature Importances / Predictive Drivers Chart.
    """
    feat_df = pd.DataFrame(top_features).sort_values(by="Importance", ascending=True)
    fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Features Associated with Churn Model Decisions",
        color="Importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        xaxis_title="Relative Feature Importance / Absolute Coefficient",
        yaxis_title="Feature",
        margin=dict(t=50, b=40, l=120, r=20),
    )
    return fig
