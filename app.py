"""
app.py - Streamlit front-end

Features:
- Auto-load datasets from /data/ (if present)
- Upload alternative CSVs
- Transaction dashboard & analyzer
- Anomaly detection
- Loan assessment form (rule-based + optional ML model)
- Chatbot (RAG) using faq_loader.FAQRetriever: OpenAI embeddings if OPENAI_API_KEY set, TF-IDF otherwise
"""

import os
import streamlit as st
import pandas as pd
from analyzer import load_transactions, detect_anomalies, kpis, assess_loan, prepare_features_for_loan, train_loan_model, load_loan_model
from faq_loader import FAQRetriever

st.set_page_config(page_title="SmartBanking System", layout="wide")
st.title("💳 SmartBanking — Loan Assessment, Transaction Analytics & RAG Chatbot")

DATA_DIR = "data"
CREDIT_CSV = os.path.join(DATA_DIR, "credit_test.csv")
BANK_CSV = os.path.join(DATA_DIR, "AI_SmartBanking_Dataset.csv")
FAQ_JSON = os.path.join(DATA_DIR, "SBI loan FAQs.json")

st.sidebar.header("Data")
uploaded_csv = st.sidebar.file_uploader("Upload transactions CSV (optional)", type=["csv"])
uploaded_faq = st.sidebar.file_uploader("Upload FAQ JSON (optional)", type=["json"])

# Load FAQ retriever (prefer uploaded -> repo file)
faq_path = None
if uploaded_faq is not None:
    # save temporarily to data/
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, "uploaded_faq.json")
    with open(p, "wb") as f:
        f.write(uploaded_faq.getbuffer())
    faq_path = p
elif os.path.exists(FAQ_JSON):
    faq_path = FAQ_JSON

faq_retriever = FAQRetriever(faq_path) if faq_path or os.path.exists(FAQ_JSON) else FAQRetriever(None)

# Tabs
tabs = st.tabs(["📊 Dashboard", "🔍 Analyzer", "🧠 Anomalies", "🏦 Loan Assessment", "🤖 RAG Chatbot"])

# Helper: load main dataframe
def load_main_df():
    if uploaded_csv is not None:
        return pd.read_csv(uploaded_csv)
    if os.path.exists(CREDIT_CSV):
        return load_transactions(CREDIT_CSV)
    if os.path.exists(BANK_CSV):
        return load_transactions(BANK_CSV)
    return pd.DataFrame()

# ---------------- Dashboard ----------------
with tabs[0]:
    st.header("Overview")
    df = load_main_df()
    if df.empty:
        st.info("No default data found. Upload a CSV or push datasets to /data/ in the repo.")
    else:
        st.dataframe(df.head(50))
        metrics = kpis(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{metrics.get('txn_count', 0):,}")
        c2.metric("Total Volume", f"{metrics.get('total_volume', 0):,.2f}")
        c3.metric("Avg. Txn", f"{metrics.get('avg_txn', 0):,.2f}")

# ---------------- Analyzer ----------------
with tabs[1]:
    st.header("Transaction Analyzer")
    df = load_main_df()
    if df.empty:
        st.info("Load data to analyze transactions.")
    else:
        st.subheader("Filters & Grouping")
        cols = list(df.columns)
        cols_select = st.multiselect("Select columns to view", cols, default=cols[:5])
        st.dataframe(df[cols_select].head(200))
        # group
        group_col = st.selectbox("Group by (optional)", ["(None)"] + cols)
        if group_col and group_col != "(None)":
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                agg_col = st.selectbox("Aggregate (numeric) column", numeric_cols)
                grouped = df.groupby(group_col)[agg_col].agg(["count", "sum", "mean"]).reset_index()
                st.dataframe(grouped)
            else:
                st.info("No numeric columns to aggregate.")

# ---------------- Anomaly Detection ----------------
with tabs[2]:
    st.header("Anomaly Detection")
    df = load_main_df()
    if df.empty:
        st.info("Load data to run anomaly detection.")
    else:
        contamination_pct = st.slider("Estimated % anomalies", 0.1, 20.0, 2.0) / 100.0
        out = detect_anomalies(df, contamination=contamination_pct)
        st.write(f"Flagged {out['anomaly'].sum() if 'anomaly' in out else 0} anomalies")
        st.dataframe(out.head(200))

# ---------------- Loan Assessment ----------------
with tabs[3]:
    st.header("Loan Assessment")

    st.markdown("### Applicant details")
    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amount = st.number_input("Loan amount (numeric)", min_value=0.0, value=500000.0)
        term = st.number_input("Term (months)", min_value=1, value=60)
        income = st.number_input("Monthly income", min_value=0.0, value=50000.0)
    with col2:
        credit_score = st.number_input("Credit score (300-900)", min_value=300, max_value=900, value=700)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        employment_years = st.number_input("Employment years", min_value=0.0, value=2.0)
    with col3:
        existing_debt = st.number_input("Existing total debt", min_value=0.0, value=0.0)
        # option to use model
        use_model = st.checkbox("Use trained loan model (if available)", value=True)

    if st.button("Assess Loan"):
        app = {
            "loan_amount": loan_amount,
            "term": term,
            "income": income,
            "credit_score": credit_score,
            "age": age,
            "employment_years": employment_years,
            "existing_debt": existing_debt,
        }
        result = assess_loan(app, use_model=use_model)
        st.markdown(f"**Decision:** {result['decision'].upper()}  ")
        st.markdown(f"**Score:** {result['score']}")
        if result.get("model_prob") is not None:
            st.markdown(f"**Model probability:** {result['model_prob']:.2f}")
        if result["reasons"]:
            st.markdown("**Reasons:**")
            for r in result["reasons"]:
                st.write("- " + r)

    st.markdown("---")
    st.markdown("### Train / Upload Loan Model (optional)")
    st.markdown("If you have a labeled loan dataset (with 'approved' or similar), upload it to train a simple model.")
    loan_train_file = st.file_uploader("Upload loan CSV for training", type=["csv"])
    if loan_train_file is not None:
        train_df = pd.read_csv(loan_train_file)
        X, y = prepare_features_for_loan(train_df)
        if y is None:
            st.error("Could not find a target column (approved/is_approved). Add a binary target column.")
        else:
            st.info(f"Training model on {X.shape[0]} rows and {X.shape[1]} features...")
            model = train_loan_model(X, y)
            st.success("Model trained and saved to models/loan_model.joblib")
            st.write("You can now use the model via the 'Use trained loan model' checkbox.")

# ---------------- RAG Chatbot ----------------
with tabs[4]:
    st.header("RAG Chatbot (FAQ)")
    st.markdown("This chatbot retrieves answers from your FAQ document and (optionally) uses an LLM to compose answers.")
    query = st.text_input("Ask a question about loans, EMI, eligibility, documents, etc.")
    if st.button("Get answer"):
        if not query.strip():
            st.info("Type a question first.")
        else:
            # retrieval
            results = faq_retriever.retrieve(query, top_k=3)
            if not results:
                st.info("No matching FAQ found. Upload a FAQ JSON to /data/ or via sidebar.")
            else:
                st.markdown("**Top retrieved FAQ(s):**")
                for q, a, score in results:
                    st.markdown(f"- **Q:** {q} (score={score:.3f})")
                    st.markdown(f"  - **A:** {a}")

                # if OpenAI is available and configured, we can optionally call it to generate a composed answer using the retrieved contexts.
                if os.environ.get("OPENAI_API_KEY"):
                    try:
                        import openai
                        openai.api_key = os.environ.get("OPENAI_API_KEY")
                        # build prompt with contexts
                        contexts = "\n\n".join([f"Q: {q}\nA: {a}" for q, a, _ in results])
                        prompt = f"Use the following FAQ snippets to answer the user's question. If the answer is not present, say you don't know.\n\nFAQ SNIPPETS:\n{contexts}\n\nUser question: {query}\n\nAnswer:"
                        resp = openai.ChatCompletion.create(
                            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                            messages=[{"role":"user","content":prompt}],
                            temperature=0.0,
                            max_tokens=512
                        )
                        answer = resp["choices"][0]["message"]["content"].strip()
                        st.markdown("**LLM-composed answer:**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"LLM generation failed: {e}")
