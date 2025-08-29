import os
import streamlit as st
import pandas as pd

# --- Import helper functions if available ---
try:
    from analyzer import load_transactions, detect_anomalies, kpis
except ImportError:
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


# --- Streamlit Config ---
st.set_page_config(page_title="AI SmartBanking", layout="wide")
st.title("💳 AI-Powered SmartBanking — Web App")


# --- Default data paths ---
DATA_DIR = "data"
CREDIT_DATA = os.path.join(DATA_DIR, "credit_test.csv")
BANK_DATA = os.path.join(DATA_DIR, "AI_SmartBanking_Dataset.csv")
FAQ_DATA = os.path.join(DATA_DIR, "SBI loan FAQs.json")


def load_default_data():
    """Try loading default datasets from /data folder."""
    if os.path.exists(CREDIT_DATA):
        st.caption("✅ Loaded default dataset: credit_test.csv")
        return load_transactions(CREDIT_DATA)
    elif os.path.exists(BANK_DATA):
        st.caption("✅ Loaded default dataset: AI_SmartBanking_Dataset.csv")
        return load_transactions(BANK_DATA)
    return None


# --- Sidebar Upload ---
st.sidebar.header("Upload Your Own Data")
uploaded_csv = st.sidebar.file_uploader("📂 Upload a transactions CSV", type=["csv"])


# --- Tabs ---
tabs = st.tabs(["📊 Dashboard", "🔍 Analyzer", "🧠 Anomalies", "🤖 Chatbot"])


# --- Dashboard ---
with tabs[0]:
    st.subheader("📊 Overview Dashboard")

    df = None
    try:
        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.caption("✅ Using uploaded file.")
        else:
            df = load_default_data()
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}")

    if df is None or df.empty:
        st.info("📂 Please upload a CSV or add datasets in `/data/` folder.")
    else:
        st.dataframe(df.head(20))

        # KPIs
        metrics = kpis(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{metrics.get('txn_count', 0):,}")
        c2.metric("Total Volume", f"{metrics.get('total_volume', 0):,.2f}")
        c3.metric("Avg. Txn", f"{metrics.get('avg_txn', 0):,.2f}")


# --- Analyzer ---
with tabs[1]:
    st.subheader("🔍 Transaction Analyzer")
    if df is None or df.empty:
        st.info("⚠️ No data loaded.")
    else:
        st.dataframe(df.head(50))


# --- Anomaly Detection ---
with tabs[2]:
    st.subheader("🧠 Anomaly Detection")
    if df is None or df.empty:
        st.info("⚠️ No data loaded.")
    else:
        anomalies = detect_anomalies(df)
        if anomalies.empty:
            st.success("✅ No anomalies found.")
        else:
            st.warning("⚠️ Potential anomalies:")
            st.dataframe(anomalies)


# --- Chatbot ---
with tabs[3]:
    st.subheader("🤖 Banking Chatbot")

    faq_bot = FAQBot(FAQ_DATA if os.path.exists(FAQ_DATA) else None)

    user_q = st.text_input("💬 Ask me a banking question:")
    if st.button("Get Answer"):
        if os.path.exists(FAQ_DATA):
            st.write("🤖:", faq_bot.answer(user_q))
        else:
            st.info("📂 FAQ file not found in `/data/`.")
