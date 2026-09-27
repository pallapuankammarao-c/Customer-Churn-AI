"""
sql_analysis.py
===============
SQLite-backed SQL analytics engine for Customer Churn data.
Loads cleaned customer records into an embedded relational database
and provides verified analytical queries, execution utilities, and business interpretations.
"""

import os
import sys
import sqlite3
from typing import Dict, Any, List, Tuple
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.data_cleaning import load_raw_data, clean_churn_data
except ImportError:
    from data_cleaning import load_raw_data, clean_churn_data

DB_PATH = "data/churn_analytics.db"
TABLE_NAME = "customers"


def init_sqlite_db(df: pd.DataFrame = None, db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Initializes SQLite database and populates the 'customers' table.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    if df is None:
        raw_df = load_raw_data()
        df, _ = clean_churn_data(raw_df)

    # Save to SQLite table
    df_to_save = df.copy()
    # Drop complex object/category types if any
    for col in df_to_save.select_dtypes(include=["category"]).columns:
        df_to_save[col] = df_to_save[col].astype(str)

    df_to_save.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    return conn


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Returns an active SQLite database connection.
    If database does not exist, it initializes it automatically.
    """
    if not os.path.exists(db_path):
        init_sqlite_db(db_path=db_path)
    return sqlite3.connect(db_path)


def run_sql_query(query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Executes a read-only SQL query against the customer churn database.
    """
    conn = get_connection(db_path)
    try:
        result_df = pd.read_sql_query(query, conn)
        return result_df
    finally:
        conn.close()


# Pre-defined professional SQL business questions
SQL_BUSINESS_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q1",
        "title": "1. Overall Customer Churn Rate",
        "question": "What is the overall churn rate and customer distribution across the entire business?",
        "query": """
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN LOWER(Churn) = 'yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN LOWER(Churn) = 'no' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(AVG(CASE WHEN LOWER(Churn) = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers;
""",
        "interpretation": "Calculates the baseline churn rate across the organization. This serves as the primary benchmark against which all customer segments are evaluated.",
    },
    {
        "id": "q2",
        "title": "2. Churn Rate by Contract Type",
        "question": "Which contract type has the highest customer churn rate?",
        "query": """
SELECT 
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN LOWER(Churn) = 'yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN LOWER(Churn) = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
""",
        "interpretation": "Identifies contract vulnerability. Month-to-month contracts exhibit significantly higher churn rates compared to one-year and two-year commitments.",
    },
    {
        "id": "q3",
        "title": "3. Average Monthly Charges: Churned vs. Retained",
        "question": "What is the average monthly charge of churned customers compared to retained customers?",
        "query": """
SELECT 
    Churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(MIN(MonthlyCharges), 2) AS min_monthly_charges,
    ROUND(MAX(MonthlyCharges), 2) AS max_monthly_charges
FROM customers
GROUP BY Churn;
""",
        "interpretation": "Highlights pricing pressure. Churned customers pay significantly higher average monthly charges than those who remain with the service.",
    },
    {
        "id": "q4",
        "title": "4. Churn by Payment Method",
        "question": "Which payment method is associated with the highest customer churn?",
        "query": """
SELECT 
    PaymentMethod,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN LOWER(Churn) = 'yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN LOWER(Churn) = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;
""",
        "interpretation": "Reveals friction in billing. Electronic check users churn at a much higher rate than customers enrolled in automatic credit card or bank transfer payments.",
    },
    {
        "id": "q5",
        "title": "5. Average Tenure of Churned vs. Retained Customers",
        "question": "What is the average tenure (in months) of churned customers compared to retained customers?",
        "query": """
SELECT 
    Churn,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    ROUND(MIN(tenure), 1) AS min_tenure_months,
    ROUND(MAX(tenure), 1) AS max_tenure_months
FROM customers
GROUP BY Churn;
""",
        "interpretation": "Demonstrates early customer life-cycle risk. Churned customers leave early in their lifecycle, while retained customers have substantially longer tenures.",
    },
    {
        "id": "q6",
        "title": "6. Churn by Internet Service Type",
        "question": "Which internet service type experiences the highest churn?",
        "query": """
SELECT 
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN LOWER(Churn) = 'yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(AVG(CASE WHEN LOWER(Churn) = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
""",
        "interpretation": "Fiber optic customers experience high churn despite generating high ARPU (average revenue per user), suggesting potential dissatisfaction with price or reliability.",
    },
    {
        "id": "q7",
        "title": "7. High-Risk Customer Profile: Month-to-Month + Electronic Check",
        "question": "What percentage of customers combine month-to-month contracts and electronic check payments, and what is their churn rate?",
        "query": """
SELECT 
    CASE 
        WHEN Contract = 'Month-to-month' AND PaymentMethod = 'Electronic check' THEN 'High-Risk Profile (M2M + E-Check)'
        ELSE 'Other Customer Profiles'
    END AS customer_profile,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS pct_of_total_base,
    SUM(CASE WHEN LOWER(Churn) = 'yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(AVG(CASE WHEN LOWER(Churn) = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS segment_churn_rate_pct
FROM customers
GROUP BY customer_profile
ORDER BY segment_churn_rate_pct DESC;
""",
        "interpretation": "Demonstrates how multi-variable SQL querying isolates acute risk segments. Customers with both Month-to-month contracts and Electronic check payment show dramatic churn rates.",
    },
    {
        "id": "q8",
        "title": "8. Revenue Impact of Customer Churn",
        "question": "What is the total monthly and lifetime revenue lost to churned customers?",
        "query": """
SELECT 
    Churn,
    COUNT(*) AS total_customers,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_revenue,
    ROUND(SUM(TotalCharges), 2) AS cumulative_lifetime_revenue
FROM customers
GROUP BY Churn;
""",
        "interpretation": "Quantifies the direct financial impact. Losing customers translates into significant lost monthly recurring revenue (MRR) and cumulative customer lifetime value.",
    },
]

# Backward and interface compatibility aliases
init_database = init_sqlite_db

def get_predefined_sql_queries() -> list[dict]:
    """Returns business questions with normalized 'sql' and 'query' keys."""
    queries = []
    for q in SQL_BUSINESS_QUESTIONS:
        item = dict(q)
        item["sql"] = q["query"]
        queries.append(item)
    return queries


if __name__ == "__main__":
    print("Initializing SQLite database...")
    init_sqlite_db()
    print("Testing SQL Queries:")
    for item in SQL_BUSINESS_QUESTIONS[:3]:
        print(f"\n--- {item['title']} ---")
        df_res = run_sql_query(item["query"])
        print(df_res)

