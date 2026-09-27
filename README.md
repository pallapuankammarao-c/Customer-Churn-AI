# 📊 AI Customer Churn Intelligence & Retention Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Portfolio-Grade Full-Stack Data Science & Analytics Application**  
> An end-to-end, production-ready analytics system that bridges technical machine learning and executive business decision-making. Predicts customer attrition, quantifies revenue at risk, investigates empirical churn drivers, and formulates data-backed customer retention strategies.

---

## 🌐 Live Web Application

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

> 🚀 **Deploy & Share:** Once pushed to your GitHub repository, this dashboard can be deployed for free on [Streamlit Community Cloud](https://share.streamlit.io/) with a single click so anyone can access it directly in their web browser! (See [Deployment Guide](#-how-to-deploy-live-on-streamlit-cloud) below).

---

## 📸 Dashboard Preview & Modules

The application is structured into **8 modular, interactive analytical suites**:

| Module | Core Purpose & Capabilities |
| :--- | :--- |
| **1. Overview** | Executive KPI cards (Total Customers, Churn Rate, MRR At Risk, High-Risk Volume), overall retention donut chart, and risk distribution. |
| **2. Customer Analysis** | Dynamic cohort filtering across demographics (Gender, Senior Citizen), contract terms, payment methods, and service types. |
| **3. Churn Drivers** | Empirical statistical breakdowns detailing the impact of month-to-month contracts, fiber optic pricing, electronic checks, and tech support. |
| **4. ML Model Suite** | Leakage-free `ColumnTransformer` pipelines, Stratified 80/20 train/test evaluation, Logistic Regression vs. Random Forest benchmarks, Confusion Matrix, and ROC-AUC curves. |
| **5. Predict Customer** | Real-time interactive churn risk simulation form. Generates attrition probability (%), risk tiers (Low/Medium/High), key risk drivers, and actionable retention strategies. |
| **6. SQL Analytics** | Built-in SQLite database engine executing analytical business queries with CTEs, plus an interactive custom SQL sandbox with read-only safety guards. |
| **7. AI Strategic Insights** | Dual-mode intelligence architecture: LLM synthesis (OpenAI-compatible) paired with an automated **Deterministic Empirical Fallback Engine** (works 100% offline without API keys). |
| **8. About & Methodology** | Project overview, technical stack documentation, and data science methodology statement. |

---

## 🎯 Key Business & Technical Highlights

- **Rigorous Data Quality Pipeline:** Handles non-trivial real-world anomalies (whitespace strings in new accounts with 0 tenure imputed to `$0.00`, automated deduplication, and schema validation).
- **Strict Zero Data Leakage:** Preprocessing transformations (StandardScaler for numerical features, OneHotEncoder for categorical features) are encapsulated strictly inside Scikit-learn `Pipeline` and `ColumnTransformer` fitted only on training splits.
- **Explainable ML Decision Support:** Translates statistical probabilities into three business risk tiers (**Low Risk < 30%**, **Medium Risk 30%–60%**, **High Risk > 60%**) with prescriptive retention tactics.
- **Enterprise Data Export:** Allows users to download cleaned datasets (CSV), model-scored risk classifications (CSV), and aggregate KPI summaries (JSON) directly from the sidebar.
- **Dual-Mode AI Insights:** Guarantees zero downtime by seamlessly falling back to a deterministic rule-based distribution engine if an external LLM API key is not present.

---

## 🏗️ Project Architecture

```plaintext
customer-churn-ai/
│
├── app.py                      # Main Streamlit application and navigation hub
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules (virtual environments, cache, etc.)
├── .env.example                # Sample environment variables for optional LLM integration
├── README.md                   # Project documentation and deployment guide
│
├── data/                       # Data storage directory
│   ├── customer_churn.csv      # IBM Telco Customer Churn benchmark dataset
│   └── churn_analytics.db      # Embedded SQLite analytical database
│
├── models/                     # Serialized machine learning models
│   └── churn_model.pkl         # Trained model pipeline artifact & evaluation telemetry
│
├── notebooks/                  # Exploratory research
│   └── exploratory_analysis.ipynb
│
└── src/                        # Modular application source code
    ├── __init__.py
    ├── data_cleaning.py        # Automated ingestion, cleaning, and quality telemetry
    ├── analysis.py             # Business KPIs and interactive Plotly chart builders
    ├── model.py                # Leakage-free training pipelines & multi-model evaluation
    ├── predictions.py          # Real-time customer inference & retention recommendations
    ├── sql_analysis.py         # SQLite analytical engine & predefined business queries
    └── ai_insights.py          # Dual-mode AI strategic synthesis engine
```

---

## 🚀 Quickstart: Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pallapuankammarao/customer-churn-ai.git
cd customer-churn-ai
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy Live on Streamlit Cloud

To make this dashboard accessible via a public web link for recruiters, portfolio reviewers, and colleagues:

1. **Push this project to your GitHub account** (see instructions below).
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository: `customer-churn-ai`.
5. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy!"**
7. Streamlit Cloud will automatically build and launch your application at a public URL like:
   `https://customer-churn-ai.streamlit.app`

---

## 📤 Push to GitHub Guide

If you haven't pushed your code to GitHub yet, run these commands in your project directory:

```bash
# 1. Initialize git
git init

# 2. Add all files
git add .

# 3. Commit files
git commit -m "feat: complete AI customer churn analytics and prediction dashboard"

# 4. Create main branch
git branch -M main

# 5. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/pallapuankammarao/customer-churn-ai.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 📊 Dataset & Benchmark Details

- **Dataset:** IBM Telco Customer Churn Benchmark
- **Sample Size:** 7,043 Customer Records
- **Feature Dimensions:** 21 Attributes across Demographics, Account Settings, Subscribed Products, and Financial Metrics.
- **Target Variable:** `Churn` (Yes / No)

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👤 Author

**Ankamma Rao Pallapu**  
- **GitHub:** [@pallapuankammarao](https://github.com/pallapuankammarao)  
- **Email:** [pallapuankamma035@gmail.com](mailto:pallapuankamma035@gmail.com)
