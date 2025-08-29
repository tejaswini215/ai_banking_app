import os
import streamlit as st
import pandas as pd

# --- Import helper functions if available ---
try:
    from analyzer import load_transactions, detect_anomalies, kpis
except ImportError:
    # Fallback dummy functions if analyzer.py not found
    def load_transactions(path):
        return pd.read_csv(path)

    def detect_anomalies(df):
        return df.sample(min(5, len(df))) if not df.empty else pd.DataFrame()

    def kpis(df):
        if df.empty:
            return {"txn_count": 0, "total_volume": 0, "avg_txn": 0}
        return {
            "txn_count": len(df),
            "total_volume": df["Amount"].sum() if "Amount" in df else 0,
            "avg_txn": df["Amount"].mean() if "Amount" in df else 0,
        }

try:
    from faq_loader import FAQBot
except ImportError:
    class FAQBot:
        def __init__(self, path=None):
            self.faq = {}

        def answer(self, query):
            return "❓ FAQ data not available."


# --- Streamlit Page Config ---
st.set_page_config(page_title="AI SmartBanking Demo", layout="wide")
st.title("💳 AI-Powered SmartBanking — Demo App")


# --- Sidebar file inputs ---
st.sidebar.header("Data Sources")

# Default paths (these may or may not exist on Streamlit Cloud)
default_credit_path = os.path.join("data", "credit_test.csv")
default_bank_path = os.path.join("data", "AI_SmartBanking_Dataset.csv")
default_faq_path = os.path.join("data", "SBI loan FAQs.json")

credit_path = st.sidebar.text_input("Path to transactions CSV",
                                    default_credit_path if os.path.exists(default_credit_path) else "")
bank_path = st.sidebar.text_input("Path to smart banking CSV",
                                  default_bank_path if os.path.exists(default_bank_path) else "")
faq_path = st.sidebar.text_input("Path to FAQ JSON",
                                 default_faq_path if os.path.exists(default_faq_path) else "")

uploaded_csv = st.sidebar.file_uploader("📂 Or upload a CSV", type=["csv"])


# --- Tabs ---
tabs = st.tabs(["📊 Dashboard", "🔍 Transaction Analyzer",
                "🧠 Anomaly Detection", "🤖 Banking Chatbot"])


# --- Dashboard ---
with tabs[0]:
    st.subheader("Overview Dashboard")

    df = None
    path_to_use = None

    try:
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            path_to_use = "(uploaded file)"
        elif credit_path and os.path.exists(credit_path):
            df = load_transactions(credit_path)
            path_to_use = credit_path
        elif bank_path and os.path.exists(bank_path):
            df = load_transactions(bank_path)
            path_to_use = bank_path
    except Exception as e:
        st.error(f"⚠️ Could not load transactions. Error: {e}")

    if df is None or df.empty:
        st.info("📂 Please upload a CSV or provide a valid path in the sidebar.")
    else:
        st.caption(f"Data source: **{path_to_use}**")
        st.dataframe(df.head(20))

        # Show metrics
        metrics = kpis(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{metrics.get('txn_count', 0):,}")
        c2.metric("Total Volume", f"{metrics.get('total_volume', 0):,.2f}")
        c3.metric("Avg. Txn", f"{metrics.get('avg_txn', 0):,.2f}")


# --- Transaction Analyzer ---
with tabs[1]:
    st.subheader("Transaction Analyzer")

    if df is None or df.empty:
        st.info("⚠️ Load or upload a transactions file first.")
    else:
        st.write("🔎 Showing first 50 transactions:")
        st.dataframe(df.head(50))


# --- Anomaly Detection ---
with tabs[2]:
    st.subheader("Anomaly Detection")

    if df is None or df.empty:
        st.info("⚠️ Load or upload a transactions file first.")
    else:
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.success("✅ No anomalies detected.")
        else:
            st.warning("⚠️ Potential anomalies found:")
            st.dataframe(anomalies)


# --- Banking Chatbot ---
with tabs[3]:
    st.subheader("Banking Chatbot (FAQs)")

    faq_bot = None
    if faq_path and os.path.exists(faq_path):
        faq_bot = FAQBot(faq_path)

    user_q = st.text_input("💬 Ask me a banking question:")
    if st.button("Get Answer"):
        if faq_bot:
            st.write("🤖:", faq_bot.answer(user_q))
        else:
            st.info("📂 FAQ file not found. Please upload or provide a valid path.")
