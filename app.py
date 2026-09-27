"""
AI-Powered Customer Churn Prediction & Analytics Dashboard
===========================================================
Portfolio-grade full-stack Data Science & Analytics Application.
Built with Streamlit, Plotly, Scikit-learn, SQLite, and Hybrid AI Insights.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Custom module imports
from src.data_cleaning import clean_customer_data, acquire_raw_data
from src.analysis import (
    compute_business_kpis,
    create_churn_donut_chart,
    create_churn_by_contract_chart,
    create_churn_by_tenure_chart,
    create_churn_by_monthly_charges_chart,
    create_churn_by_payment_method_chart,
    create_churn_by_internet_service_chart,
    create_churn_by_senior_citizen_chart,
    create_churn_by_tech_support_chart,
    create_customer_risk_distribution_chart,
    CHURN_COLOR,
    RETAIN_COLOR,
    ACCENT_GREEN,
    ACCENT_AMBER
)
from src.model import load_trained_model, train_and_evaluate_models, NUMERICAL_COLS, CATEGORICAL_COLS
from src.predictions import predict_single_customer, score_entire_dataset, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from src.sql_analysis import init_database, run_sql_query, get_predefined_sql_queries
from src.ai_insights import get_ai_business_insights, build_aggregated_data_summary

# -------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Global Typography & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* KPI Cards */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        flex: 1 1 180px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* Risk Badges */
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }

    /* Section Headers */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .dashboard-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 24px;
    }
    .section-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 24px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# DATA CACHING & INITIALIZATION
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare_app_data():
    """Loads, cleans, initializes SQLite, and scores risk tiers."""
    df, quality_report = clean_customer_data()
    # Initialize SQLite database
    init_database(df)
    # Load model and score risk tiers
    model, metrics = load_trained_model(df)
    scored_df = score_entire_dataset(df, model)
    return df, quality_report, scored_df, model, metrics


# Load core data assets
with st.spinner("Initializing Customer Churn Intelligence System..."):
    df, quality_report, scored_df, model, model_metrics = load_and_prepare_app_data()


# -------------------------------------------------------------
# SIDEBAR NAVIGATION & DATA EXPORT
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/analytics.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1. Overview",
        "2. Customer Analysis",
        "3. Churn Drivers",
        "4. ML Model",
        "5. Predict Customer",
        "6. SQL Analysis",
        "7. AI Insights",
        "8. About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Export")

# Export Cleaned Data
csv_clean = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Cleaned CSV",
    data=csv_clean,
    file_name="cleaned_customer_churn.csv",
    mime="text/csv",
    use_container_width=True
)

# Export Scored Risk Predictions
csv_scored = scored_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Scored Predictions",
    data=csv_scored,
    file_name="customer_churn_scored_risk.csv",
    mime="text/csv",
    use_container_width=True
)

# Export Analysis Summary JSON
summary_dict = build_aggregated_data_summary(df)
json_summary = json.dumps(summary_dict, indent=2).encode("utf-8")
st.sidebar.download_button(
    label="Download Summary JSON",
    data=json_summary,
    file_name="churn_summary_metrics.json",
    mime="application/json",
    use_container_width=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio Project by CS Undergraduate\nDual Rule-Based & LLM AI Architecture")


# -------------------------------------------------------------
# HELPER: RENDER KPI CARDS
# -------------------------------------------------------------
def render_kpi_banner(kpis: dict):
    """Renders high-impact KPI summary cards at the top of pages."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
            <div class="kpi-subtext">Active cohort</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Churned Volume</div>
            <div class="kpi-value" style="color: {CHURN_COLOR};">{kpis['churn_count']:,}</div>
            <div class="kpi-subtext">Terminated service</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value" style="color: {CHURN_COLOR};">{kpis['churn_rate']}%</div>
            <div class="kpi-subtext">Attrition ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Monthly Spend</div>
            <div class="kpi-value">${kpis['avg_monthly_charges']:.2f}</div>
            <div class="kpi-subtext">ARPU per month</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Tenure</div>
            <div class="kpi-value">{kpis['avg_tenure']} <span style="font-size:1rem;font-weight:500;">mos</span></div>
            <div class="kpi-subtext">Customer lifespan</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High-Risk Base</div>
            <div class="kpi-value" style="color: #DC2626;">{kpis['high_risk_count']:,}</div>
            <div class="kpi-subtext">{kpis['high_risk_pct']}% of portfolio</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# PAGE 1: OVERVIEW
# =============================================================
if page == "1. Overview":
    st.markdown('<div class="dashboard-title">AI-Powered Customer Churn Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Data-driven insights for understanding, quantifying, and predicting customer attrition</div>', unsafe_allow_html=True)

    kpis = compute_business_kpis(df, scored_df)
    render_kpi_banner(kpis)

    # Business Summary Callout
    # Calculate key segment highlights dynamically
    contract_churn = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).to_dict()
    m2m_rate = contract_churn.get("Month-to-month", 0.0)
    top_contract = max(contract_churn.items(), key=lambda x: x[1])[0]

    st.markdown(f"""
    <div class="section-card">
        <h4 style="margin-top:0; color:#0F172A;">Executive Business Summary</h4>
        <ul style="color:#334155; font-size:0.95rem; line-height:1.6; margin-bottom:0;">
            <li><b>Current Portfolio Churn Rate:</b> <code>{kpis['churn_rate']}%</code> across {kpis['total_customers']:,} observed accounts.</li>
            <li><b>Revenue Impact:</b> Attrition currently accounts for <b>${kpis['churn_monthly_revenue_loss']:,.2f}</b> in lost monthly recurring revenue.</li>
            <li><b>Customers Requiring Attention:</b> <b>{kpis['high_risk_count']:,} customers</b> ({kpis['high_risk_pct']}% of base) fall into the high-risk classification tier (predicted probability &gt; 60%).</li>
            <li><b>Highest Observed Churn Segment:</b> <code>{top_contract}</code> contract holders exhibit the highest attrition rate at <b>{m2m_rate:.1f}%</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Core Charts Grid
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_churn_donut_chart(df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(create_customer_risk_distribution_chart(scored_df), use_container_width=True)

    st.plotly_chart(create_churn_by_contract_chart(df), use_container_width=True)

    # Data Quality & Ingestion Telemetry
    with st.expander("🔍 View Data Quality & Preprocessing Telemetry", expanded=False):
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        qcol1.metric("Raw Rows Processed", f"{quality_report['raw_rows']:,}")
        qcol2.metric("Duplicates Removed", f"{quality_report['duplicates_removed']:,}")
        qcol3.metric("Whitespace Fixes (TotalCharges)", f"{quality_report['whitespace_total_charges_fixed']:,}")
        qcol4.metric("Post-Cleaning Missing Values", f"{quality_report['clean_missing_values']:,}")
        st.caption("Preprocessing Pipeline: Missing charges for 0-tenure accounts imputed to $0.00. Categoricals standardized. No lossy row deletion.")


# =============================================================
# PAGE 2: CUSTOMER ANALYSIS
# =============================================================
elif page == "2. Customer Analysis":
    st.markdown('<div class="dashboard-title">Customer Segmentation & Cohort Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Filter cohorts across demographic, contractual, and service dimensions</div>', unsafe_allow_html=True)

    # Filter Controls
    with st.expander("Filter Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            contract_filter = st.multiselect("Contract Type", options=df["Contract"].unique(), default=df["Contract"].unique())
            internet_filter = st.multiselect("Internet Service", options=df["InternetService"].unique(), default=df["InternetService"].unique())
        with fcol2:
            payment_filter = st.multiselect("Payment Method", options=df["PaymentMethod"].unique(), default=df["PaymentMethod"].unique())
            gender_filter = st.multiselect("Gender", options=df["gender"].unique(), default=df["gender"].unique())
        with fcol3:
            senior_filter = st.multiselect("Senior Citizen", options=[0, 1], format_func=lambda x: "Senior Citizen" if x == 1 else "Non-Senior", default=[0, 1])
            churn_filter = st.multiselect("Churn Status", options=["All", "Yes", "No"], default=["All"])

    # Apply Filters
    filtered_df = df.copy()
    if contract_filter:
        filtered_df = filtered_df[filtered_df["Contract"].isin(contract_filter)]
    if internet_filter:
        filtered_df = filtered_df[filtered_df["InternetService"].isin(internet_filter)]
    if payment_filter:
        filtered_df = filtered_df[filtered_df["PaymentMethod"].isin(payment_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df["gender"].isin(gender_filter)]
    if senior_filter:
        filtered_df = filtered_df[filtered_df["SeniorCitizen"].isin(senior_filter)]
    if "All" not in churn_filter and churn_filter:
        filtered_df = filtered_df[filtered_df["Churn"].isin(churn_filter)]

    # Filtered KPI Banner
    filtered_kpis = compute_business_kpis(filtered_df)
    st.write("### Filtered Cohort Performance")
    render_kpi_banner(filtered_kpis)

    # Comparative Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_churn_by_tenure_chart(filtered_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_churn_by_monthly_charges_chart(filtered_df), use_container_width=True)

    # Customer Data Table Preview
    st.write("### Detailed Customer Records (Filtered Cohort)")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Displaying top {min(100, len(filtered_df))} of {len(filtered_df):,} matching customer accounts.")


# =============================================================
# PAGE 3: CHURN DRIVERS
# =============================================================
elif page == "3. Churn Drivers":
    st.markdown('<div class="dashboard-title">Empirical Churn Drivers & Behavioral Patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Investigating statistical relationships between customer attributes and attrition</div>', unsafe_allow_html=True)

    # Driver 1 & 2
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(create_churn_by_contract_chart(df), use_container_width=True)
        st.info("Observed Pattern: Month-to-month contracts experience over 40% attrition, compared to under 3% for two-year agreements. Short commitment periods expose customers to frequent switching triggers.")
    with c2:
        st.plotly_chart(create_churn_by_internet_service_chart(df), use_container_width=True)
        st.info("Observed Pattern: Fiber optic accounts show the highest churn rate (~41%), while DSL remains under 20%. This highlights potential onboarding or technical support friction in premium tiers.")

    # Driver 3 & 4
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(create_churn_by_payment_method_chart(df), use_container_width=True)
        st.info("Observed Pattern: Electronic check users churn at nearly double the rate of customers utilizing automated credit card or bank transfer billing channels.")
    with c4:
        st.plotly_chart(create_churn_by_senior_citizen_chart(df), use_container_width=True)
        st.info("Observed Pattern: Senior citizens exhibit higher churn rates (~41% vs ~23%), suggesting opportunities for dedicated assistance, simplified billing, and targeted care programs.")

    # Driver 5: Add-ons
    st.plotly_chart(create_churn_by_tech_support_chart(df), use_container_width=True)
    st.info("Observed Pattern: Bundling Tech Support and Online Security correlates with a reduction in churn from ~40%+ down to ~15%. Active support creates substantial service stickiness.")


# =============================================================
# PAGE 4: ML MODEL
# =============================================================
elif page == "4. ML Model":
    st.markdown('<div class="dashboard-title">Supervised Machine Learning Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Strict leakage-free training, pipeline preprocessing, and multi-model benchmarking</div>', unsafe_allow_html=True)

    # Leakage Prevention Card
    st.markdown("""
    <div class="section-card">
        <h4 style="margin-top:0; color:#0F172A;">🛡️ Rigorous Methodology: Zero Data Leakage</h4>
        <p style="color:#475569; font-size:0.92rem; line-height:1.5;">
            In production machine learning, fitting scalers or encoders on the complete dataset before splitting allows information from the test set to leak into training, inflating metrics artificially.
            Here, the dataset is first partitioned into <b>80% Training</b> and <b>20% Testing</b> cohorts via <code>StratifiedShuffleSplit</code>.
            All feature transformations (StandardScaler for numericals, OneHotEncoder for categoricals) are encapsulated inside a Scikit-learn <code>ColumnTransformer</code> and fitted <b>strictly on the training split</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Samples breakdown
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Training Set Size (80%)", f"{model_metrics['train_samples']:,} rows")
    scol2.metric("Unseen Test Set Size (20%)", f"{model_metrics['test_samples']:,} rows")
    scol3.metric("Selected Production Model", model_metrics['best_model_name'])

    # Model Comparison Table
    st.write("### Model Performance Comparison (Evaluated on Unseen Test Set)")
    comp_data = []
    for mname, mdata in model_metrics["models"].items():
        comp_data.append({
            "Model": mname,
            "Accuracy": f"{mdata['accuracy'] * 100:.2f}%",
            "Precision": f"{mdata['precision'] * 100:.2f}%",
            "Recall": f"{mdata['recall'] * 100:.2f}%",
            "F1-Score": f"{mdata['f1'] * 100:.2f}%",
            "ROC-AUC": f"{mdata['roc_auc']:.4f}",
            "Selected": " Yes" if mname == model_metrics["best_model_name"] else "No"
        })
    comp_df = pd.DataFrame(comp_data)
    st.table(comp_df)

    # Confusion Matrix & ROC Curve
    m_col1, m_col2 = st.columns(2)
    selected_model_metrics = model_metrics["models"][model_metrics["best_model_name"]]

    with m_col1:
        st.write("#### Confusion Matrix (Selected Model)")
        cm = np.array(selected_model_metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=["Stay (0)", "Churn (1)"],
            y=["Stay (0)", "Churn (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(title="Confusion Matrix on Test Split", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with m_col2:
        st.write("#### ROC Curve Comparison")
        fig_roc = go.Figure()
        # Diagonal reference
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess (AUC = 0.50)", line=dict(dash="dash", color="gray")))
        for mname, mdata in model_metrics["models"].items():
            fig_roc.add_trace(go.Scatter(
                x=mdata["fpr"],
                y=mdata["tpr"],
                mode="lines",
                name=f"{mname} (AUC = {mdata['roc_auc']:.3f})"
            ))
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=20, r=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Beginner Friendly Metric Definitions
    with st.expander("📚 Guide to Evaluation Metrics (Beginner & Interview Friendly)"):
        st.markdown("""
        - **Accuracy:** The percentage of total predictions that were correct. *Caution:* With imbalanced churn data (e.g. 73% stay), a naive model predicting 'Stay' for everyone gets 73% accuracy while catching zero churners!
        - **Recall (Sensitivity):** Of all actual churners, what percentage did the model catch? High recall is essential in customer retention because missing a churner costs real revenue.
        - **Precision:** Of all customers the model flagged as churners, how many actually churned? High precision prevents wasting marketing budgets on loyal customers.
        - **F1-Score:** The harmonic mean of Precision and Recall. The gold standard for imbalanced classification problems.
        - **ROC-AUC:** Measures the model's ability to rank customers correctly across all classification thresholds from 0 to 1.
        """)

    # Retrain Model Button
    if st.button("🔄 Retrain Models on Latest Data"):
        with st.spinner("Retraining pipelines with stratified cross-validation..."):
            new_metrics = train_and_evaluate_models(df)
            st.success(f"Models successfully retrained! Best performer: {new_metrics['best_model_name']}")
            st.rerun()


# =============================================================
# PAGE 5: PREDICT CUSTOMER
# =============================================================
elif page == "5. Predict Customer":
    st.markdown('<div class="dashboard-title">Individual Customer Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Simulate real-time customer profiling, churn probability scoring, and retention routing</div>', unsafe_allow_html=True)

    with st.form("customer_input_form"):
        st.subheader("1. Account Demographics & Status")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with dcol2:
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with dcol3:
            partner = st.selectbox("Partner", ["Yes", "No"])
        with dcol4:
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("2. Contract & Billing Profile")
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=4)
        with bcol2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with bcol3:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
        with bcol4:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        ccol1, ccol2 = st.columns(2)
        with ccol1:
            monthly_charges = st.number_input("Monthly Charges ($ USD)", min_value=18.0, max_value=150.0, value=85.50, step=0.5)
        with ccol2:
            # Estimate total charges automatically based on tenure
            est_total = round(max(float(monthly_charges), float(tenure * monthly_charges)), 2)
            total_charges = st.number_input("Total Charges ($ USD)", min_value=0.0, max_value=10000.0, value=est_total, step=5.0)

        st.subheader("3. Subscribed Products & Services")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with scol2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with scol3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submit_btn = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

    if submit_btn:
        customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        result = predict_single_customer(customer_dict, model, df)

        st.markdown("---")
        st.write("### Prediction Results & Risk Diagnostics")

        # Top Result Callout
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Predicted Status", result["prediction"])
        with pcol2:
            st.metric("Churn Probability", f"{result['churn_probability_pct']}%")
        with pcol3:
            risk_tier = result["risk_category"]
            badge_class = "badge-high" if risk_tier == "High Risk" else ("badge-med" if risk_tier == "Medium Risk" else "badge-low")
            st.markdown(f"<div style='margin-top:14px;'><span class='{badge_class}' style='font-size:1.1rem;'>{risk_tier}</span></div>", unsafe_allow_html=True)

        # Risk Factors & Retention Recommendations
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown("""
            <div class="section-card">
                <h4 style="margin-top:0; color:#0F172A;">Associated Risk Factors</h4>
                <ul style="color:#334155; font-size:0.92rem; line-height:1.6;">
            """, unsafe_allow_html=True)
            for factor in result["contributing_factors"]:
                st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
            st.markdown("""
                </ul>
                <p style="font-size:0.8rem; color:#64748B; margin-top:8px;">
                    *Note: Contributing factors are empirical associations learned by the model and do not constitute individual causal proof.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with rcol2:
            st.markdown("""
            <div class="section-card">
                <h4 style="margin-top:0; color:#0F172A;">Recommended Retention Actions</h4>
                <ul style="color:#334155; font-size:0.92rem; line-height:1.6;">
            """, unsafe_allow_html=True)
            for rec in result["recommendations"]:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown("""
                </ul>
                <p style="font-size:0.8rem; color:#64748B; margin-top:8px;">
                    *Note: Recommendations are decision-support suggestions for customer success teams, not guaranteed outcomes.
                </p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# PAGE 6: SQL ANALYSIS
# =============================================================
elif page == "6. SQL Analysis":
    st.markdown('<div class="dashboard-title">SQL Analytical Query Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Enterprise database querying using SQLite to extract strategic business KPIs</div>', unsafe_allow_html=True)

    queries = get_predefined_sql_queries()
    query_titles = [q["title"] for q in queries]
    selected_query_title = st.selectbox("Select Business Analysis Question", query_titles)

    selected_query = next(q for q in queries if q["title"] == selected_query_title)

    st.write(f"### {selected_query['question']}")
    
    # Show SQL query
    st.code(selected_query["sql"], language="sql")

    # Run query and display results
    try:
        result_df = run_sql_query(selected_query["sql"])
        st.dataframe(result_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error executing SQL: {e}")

    st.markdown(f"""
    <div class="section-card">
        <h5 style="margin-top:0; color:#0F172A;">Business Interpretation</h5>
        <p style="color:#334155; margin-bottom:0;">{selected_query['interpretation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Custom SQL Workspace
    st.markdown("---")
    st.write("### 💻 Custom SQL Sandbox")
    st.caption("Write and execute custom read-only SQL queries against the <code>customers</code> table.")
    
    default_custom_sql = "SELECT Contract, InternetService, COUNT(*) AS count, ROUND(AVG(MonthlyCharges), 2) AS avg_charges FROM customers GROUP BY Contract, InternetService ORDER BY count DESC LIMIT 10;"
    custom_sql = st.text_area("Custom SQL Query", value=default_custom_sql, height=100)

    if st.button("Run Custom SQL"):
        # Simple read-only guard
        if any(keyword in custom_sql.upper() for keyword in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]):
            st.error("Operation restricted: Only SELECT read-only queries are permitted in this sandbox.")
        else:
            try:
                custom_result = run_sql_query(custom_sql)
                st.dataframe(custom_result, use_container_width=True)
                st.caption(f"Returned {len(custom_result)} rows.")
            except Exception as ex:
                st.error(f"SQL Syntax Error: {ex}")


# =============================================================
# PAGE 7: AI INSIGHTS
# =============================================================
elif page == "7. AI Insights":
    st.markdown('<div class="dashboard-title">AI Strategic Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Executive-ready intelligence powered by dual-mode architecture (LLM + Fallback Empirical Engine)</div>', unsafe_allow_html=True)

    # Informational architecture banner
    st.markdown("""
    <div class="section-card">
        <h5 style="margin-top:0; color:#0F172A;">🤖 Hybrid AI Architecture</h5>
        <p style="color:#475569; font-size:0.92rem; margin-bottom:0;">
            This system integrates a production-grade <b>Dual-Mode Intelligence Engine</b>:
            <br>1. If an OpenAI-compatible API key is supplied via <code>.env</code>, it transmits privacy-preserved aggregate KPIs to an LLM for dynamic narrative generation.
            <br>2. If no API key is detected or network is restricted, the built-in <b>Deterministic Empirical AI Engine</b> immediately calculates insights directly from empirical distributions. The application <b>never crashes or requires a paid subscription</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡ Generate AI Strategic Insights", type="primary", use_container_width=True):
        with st.spinner("Analyzing cross-sectional portfolio distributions and synthesizing strategic recommendations..."):
            insights = get_ai_business_insights(df)
            st.session_state["cached_insights"] = insights

    # Check for insights in session state or generate default
    if "cached_insights" not in st.session_state:
        st.session_state["cached_insights"] = get_ai_business_insights(df)

    insights = st.session_state["cached_insights"]

    st.markdown(f"**Generation Engine:** `{insights.get('engine', 'Empirical Analytics Engine')}`")
    if "notice" in insights:
        st.warning(insights["notice"])

    # 1. Executive Summary
    st.markdown("### 1. Executive Summary")
    st.info(insights.get("executive_summary", ""))

    # 2. Key Observations
    st.markdown("### 2. Main Churn Patterns & Observations")
    for obs in insights.get("key_observations", []):
        st.markdown(f"- {obs}")

    # 3. High Risk Profile
    st.markdown("### 3. High-Risk Customer Profile")
    st.markdown(f"> {insights.get('high_risk_profile', '')}")

    # 4. Business Factors
    st.markdown("### 4. Underlying Business Factors & Impact")
    for factor in insights.get("business_factors", []):
        st.markdown(f"- {factor}")

    # 5. Recommended Actions
    st.markdown("### 5. Recommended Retention Actions")
    actions = insights.get("recommended_actions", [])
    if isinstance(actions, list):
        action_data = []
        for a in actions:
            if isinstance(a, dict):
                action_data.append({
                    "Action": a.get("action", ""),
                    "Priority": a.get("priority", "Medium"),
                    "Timeframe": a.get("timeframe", ""),
                    "Strategic Detail": a.get("detail", "")
                })
            else:
                action_data.append({"Action": str(a), "Priority": "Medium", "Timeframe": "Near term", "Strategic Detail": ""})
        st.table(pd.DataFrame(action_data))

    # 6. Strategic Questions
    st.markdown("### 6. Strategic Questions for Further Investigation")
    for q in insights.get("strategic_questions", []):
        st.markdown(f"- ❓ {q}")

    # 7. Methodological Limitations
    st.markdown("### 7. Analytical Limitations & Correlation vs. Causation")
    st.caption(insights.get("methodological_limitations", ""))


# =============================================================
# PAGE 8: ABOUT PROJECT
# =============================================================
elif page == "8. About Project":
    st.markdown('<div class="dashboard-title">About This Portfolio Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">A comprehensive demonstration of full-stack data analytics, machine learning, and business intelligence</div>', unsafe_allow_html=True)

    acol1, acol2 = st.columns([2, 1])

    with acol1:
        st.markdown("""
        ### Project Overview
        This application was engineered by a third-year Computer Science student to demonstrate end-to-end data analytics and applied machine learning competencies. 
        Rather than presenting an isolated Jupyter Notebook or a static slide deck, this project delivers an interactive, production-ready analytics portal that directly bridges technical machine learning and executive business decision-making.

        ### Key Technical Highlights
        - **Data Pipeline:** Strict data cleaning handling messy types and edge cases (e.g. whitespace values in new accounts).
        - **Zero Data Leakage:** Preprocessing transformations fitted strictly on training data using Scikit-learn `Pipeline` and `ColumnTransformer`.
        - **Model Comparison:** Comprehensive benchmark between Logistic Regression and Random Forest using balanced class weights, F1-scores, and ROC-AUC.
        - **Enterprise SQL Engine:** Built-in SQLite database answering core analytical business questions with structured CTEs and indexes.
        - **Interactive UI/UX:** Responsive Plotly visualizations with tailored color palettes, KPI cards, and custom CSS styling.
        - **Hybrid AI Insights:** Deterministic fallback engine combined with LLM integration to guarantee reliability without external API dependencies.
        """)

    with acol2:
        st.markdown("""
        <div class="section-card">
            <h4 style="margin-top:0; color:#0F172A;">Technical Stack</h4>
            <ul style="color:#334155; font-size:0.9rem; line-height:1.6;">
                <li><b>Language:</b> Python 3.10+</li>
                <li><b>Analytics:</b> Pandas, NumPy</li>
                <li><b>Visualizations:</b> Plotly Express / Graph Objects</li>
                <li><b>ML Framework:</b> Scikit-learn (Pipelines, Ensembles)</li>
                <li><b>Database:</b> SQLite3</li>
                <li><b>Interface:</b> Streamlit</li>
                <li><b>Persistence:</b> Joblib</li>
                <li><b>Environment:</b> Python-dotenv</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Developed for Data Analyst / Machine Learning Engineer Portfolio Interviews. Licensed under MIT.")
