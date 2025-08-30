import os
import streamlit as st
import pandas as pd
import plotly.express as px
from analyzer import load_transactions, detect_anomalies, kpis, assess_loan, prepare_features_for_loan, train_loan_model
from faq_loader import FAQRetriever

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SmartBanking", layout="wide")

# Branding
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem;}
        .stMetric {background: #f9f9f9; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: 600; padding: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 SmartBanking Portal")

DATA_DIR = "data"
CREDIT_CSV = os.path.join(DATA_DIR, "credit_test.csv")
BANK_CSV = os.path.join(DATA_DIR, "AI_SmartBanking_Dataset.csv")
FAQ_JSON = os.path.join(DATA_DIR, "SBI loan FAQs.json")

df = load_transactions("data/AI_SmartBanking_Dataset.csv")
faq_retriever = FAQRetriever("data/SBI loan FAQs.json")

# Load FAQ retriever
def load_main_df():
    if os.path.exists("data/AI_SmartBanking_Dataset.csv"):
        return load_transactions("data/AI_SmartBanking_Dataset.csv")
    elif os.path.exists("data/credit_test.csv"):
        return load_transactions("data/credit_test.csv")
    return pd.DataFrame()

# --- FAQ Retriever ---
faq_path = None
if os.path.exists("data/SBI loan FAQs.json"):
    faq_path = "data/SBI loan FAQs.json"

faq_retriever = FAQRetriever(faq_path) if faq_path else FAQRetriever(None)
# -------------------- NAVIGATION --------------------
tabs = st.tabs(
    ["🏠 Dashboard", "📊 Transactions", "🏦 Loan Assessment", "🤖 Banking Chatbot"]
)

# -------------------- DASHBOARD --------------------
with tabs[0]:
    st.subheader("Banking Overview")

    df = load_main_df()
    if df.empty:
        st.info("Upload your transactions from the sidebar to see insights.")
    else:
        metrics = kpis(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{metrics['txn_count']:,}")
        c2.metric("Total Volume", f"₹{metrics['total_volume']:,.2f}")
        c3.metric("Average Transaction", f"₹{metrics['avg_txn']:,.2f}")

        # Chart: distribution
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            fig = px.histogram(df, x=num_cols[0], nbins=30, title="Transaction Distribution")
            st.plotly_chart(fig, use_container_width=True)


# -------------------- TRANSACTIONS --------------------
with tabs[1]:
    st.subheader("Transaction Analytics")

    df = load_main_df()
    if df.empty:
        st.info("Upload a transaction CSV to analyze patterns.")
    else:
        contamination_pct = st.slider("Anomaly % sensitivity", 0.1, 20.0, 2.0) / 100.0
        out = detect_anomalies(df, contamination=contamination_pct)

        st.metric("Detected Anomalies", f"{out['anomaly'].sum()}")

        # Chart anomalies
        if "amount" in out.columns:
            fig = px.scatter(
                out,
                x=out.index,
                y="amount",
                color="anomaly",
                title="Anomalous Transactions",
                labels={"x": "Transaction Index", "amount": "Amount"},
            )
            st.plotly_chart(fig, use_container_width=True)


# -------------------- LOAN ASSESSMENT --------------------
with tabs[2]:
    st.subheader("Loan Eligibility Assessment")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=500000.0)
        income = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
        term = st.number_input("Term (months)", min_value=1, value=60)
    with col2:
        credit_score = st.number_input("Credit Score (300-900)", 300, 900, 700)
        employment_years = st.number_input("Employment Years", min_value=0, value=2)
        existing_debt = st.number_input("Existing Debt", min_value=0.0, value=0.0)
    with col3:
        age = st.number_input("Age", 18, 100, 30)

    if st.button("Check Eligibility"):
        app = {
            "loan_amount": loan_amount,
            "term": term,
            "income": income,
            "credit_score": credit_score,
            "age": age,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
        }
        result = assess_loan(app)

        st.success(f"Decision: **{result['decision'].upper()}**")
        st.info(f"Score: {result['score']}")
        if result.get("model_prob") is not None:
            st.info(f"ML Model Confidence: {result['model_prob']:.2f}")
        if result["reasons"]:
            st.warning("Reasons: " + ", ".join(result["reasons"]))


# -------------------- CHATBOT --------------------
with tabs[3]:
    st.subheader("Smart Banking Chatbot")

    query = st.text_input("Ask me about loans, EMI, documents, eligibility...")
    if st.button("Get Answer"):
        if not query.strip():
            st.info("Please enter a question.")
        else:
            results = faq_retriever.retrieve(query, top_k=2)
            if not results:
                st.error("No FAQ data available. Upload SBI FAQs JSON.")
            else:
                best_q, best_a, score = results[0]
                st.write(f"**Answer:** {best_a}")
                st.caption(f"(Matched FAQ: {best_q}, score={score:.2f})")


# ----------------- Load Dataset -----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/b46c8a0e-7ba4-48a4-912c-2af0d25308ad.xlsx")
        # Standardize column names for easier use
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()

# ----------------- Public KPIs -----------------
st.title("📊 SmartBanking – Transaction Overview")

if not df.empty:
    st.metric("Total Transactions", len(df))
    if "amount" in df.columns:
        st.metric("Total Volume", f"₹{df['amount'].sum():,.2f}")
    if "amount" in df.columns:
        st.metric("Average Transaction", f"₹{df['amount'].mean():,.2f}")

# ----------------- Restricted History -----------------
st.subheader("🔒 Transaction History (Restricted Access)")

access_code = st.text_input("Enter Access Code", type="password")

if st.button("Unlock History"):
    if access_code == "BANK123":   # you can replace with secure auth later
        st.success("✅ Access Granted: Showing your transaction history")

        # Display cleaned transaction table
        if "date" in df.columns and "amount" in df.columns:
            st.dataframe(df[["date", "amount", "type", "merchant"]].head(50))
        else:
            st.dataframe(df.head(50))
    else:
        st.error("❌ Invalid Access Code. Permission Denied.")
