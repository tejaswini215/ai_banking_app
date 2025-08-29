"""
analyzer.py

Provides:
- load_transactions(path)
- detect_anomalies(df, contamination)
- kpis(df)
- prepare_features_for_loan(df)  # helper to prepare training features
- train_loan_model(X, y)         # train a simple model (optional)
- assess_loan(application_dict)  # rule-based assessment + optional ML score
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

LOAN_MODEL_PATH = os.path.join("models", "loan_model.joblib")

def load_transactions(path: str) -> pd.DataFrame:
    """Load CSV path to DataFrame. Tries to parse dates and normalize columns."""
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # try parse any date-like columns
    for col in df.columns:
        if col.lower() in ("date", "transaction_date", "timestamp", "time"):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42):
    """IsolationForest on numeric columns or length-encoded strings if no numeric."""
    if df is None or df.empty:
        return pd.DataFrame()
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] == 0:
        # simple encoding: length of text per column
        num_df = df.fillna("").astype(str).applymap(len)
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    model = IsolationForest(
        n_estimators=200,
        contamination=min(max(contamination, 0.001), 0.25),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(num_df.values)
    scores = model.decision_function(num_df.values)
    preds = model.predict(num_df.values)  # -1 anomaly, 1 normal
    out = df.copy()
    out["anomaly_score"] = scores
    out["anomaly"] = preds == -1
    return out

def kpis(df: pd.DataFrame):
    """Basic KPIs: txn_count, total_volume, avg_txn, top merchants (if available)."""
    out = {"txn_count": 0, "total_volume": 0.0, "avg_txn": 0.0}
    if df is None or df.empty:
        return out
    out["txn_count"] = int(df.shape[0])
    # pick amount column heuristically
    amt_col = None
    for cand in ("amount", "Amount", "transaction_amount", "amt"):
        if cand in df.columns:
            amt_col = cand
            break
    if amt_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            amt_col = num_cols[0]
    if amt_col is not None and amt_col in df.columns:
        out["total_volume"] = float(df[amt_col].sum())
        out["avg_txn"] = float(df[amt_col].mean())
    # optional extras
    if "merchant" in df.columns:
        out["top_merchants"] = df["merchant"].value_counts().head(5).to_dict()
    return out

# ---------------- Loan assessment helpers ----------------

def prepare_features_for_loan(df: pd.DataFrame):
    """
    Minimal feature extraction for loan model training.
    Assumes df contains columns like: 'loan_amount', 'term', 'income', 'age',
    'credit_score', 'existing_debt', 'employment_years', and 'approved' (target).
    Returns X (DataFrame) and y (Series) if possible.
    """
    candidates = df.copy()
    # common names mapping
    cmap = {c.lower(): c for c in candidates.columns}
    def get(col_names):
        for name in col_names:
            if name in cmap:
                return cmap[name]
        return None

    loan_col = get(["loan_amount", "amount", "loan"])
    income_col = get(["income", "monthly_income", "annual_income"])
    cs_col = get(["credit_score", "creditscore"])
    age_col = get(["age"])
    debt_col = get(["existing_debt", "debt"])
    term_col = get(["term", "tenure"])
    emp_col = get(["employment_years", "emp_years", "years_employed"])
    y_col = get(["approved", "is_approved", "target"])

    features = pd.DataFrame()
    if loan_col: features["loan_amount"] = pd.to_numeric(candidates[loan_col], errors="coerce").fillna(0)
    if income_col: features["income"] = pd.to_numeric(candidates[income_col], errors="coerce").fillna(0)
    if cs_col: features["credit_score"] = pd.to_numeric(candidates[cs_col], errors="coerce").fillna(600)
    if age_col: features["age"] = pd.to_numeric(candidates[age_col], errors="coerce").fillna(35)
    if debt_col: features["existing_debt"] = pd.to_numeric(candidates[debt_col], errors="coerce").fillna(0)
    if term_col: features["term"] = pd.to_numeric(candidates[term_col], errors="coerce").fillna(12)
    if emp_col: features["employment_years"] = pd.to_numeric(candidates[emp_col], errors="coerce").fillna(1)

    if y_col:
        y = candidates[y_col].apply(lambda v: 1 if str(v).strip().lower() in ("1","true","yes","approved") else 0)
        return features, y
    return features, None

def train_loan_model(X: pd.DataFrame, y: pd.Series):
    """Train a basic LogisticRegression model and persist if models/ exists."""
    if X is None or X.empty or y is None:
        raise ValueError("X/y must be provided to train model.")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X.fillna(0), y)
    os.makedirs(os.path.dirname(LOAN_MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, LOAN_MODEL_PATH)
    return pipe

def load_loan_model():
    """Load persisted loan model if available."""
    if os.path.exists(LOAN_MODEL_PATH):
        return joblib.load(LOAN_MODEL_PATH)
    return None

def assess_loan(application: dict, use_model: bool = True):
    """
    Rule-based scoring + optional model score.

    application: dict with keys like loan_amount, income, credit_score, age, employment_years, existing_debt.
    Returns: dict with score, decision ('approve'|'review'|'deny'), and reasons.
    """
    reasons = []
    score = 0.0

    # extract inputs with default fallbacks
    loan_amount = float(application.get("loan_amount", 0))
    income = float(application.get("income", 0))
    credit_score = float(application.get("credit_score", 600))
    age = float(application.get("age", 35))
    employment_years = float(application.get("employment_years", 1))
    existing_debt = float(application.get("existing_debt", 0))

    # simple rules
    # affordability ratio: loan_amount / (income * 12) (annualized)
    affordability = loan_amount / (max(income, 1) * 12)
    if affordability < 0.5:
        score += 40
    elif affordability < 1.0:
        score += 20
    else:
        score -= 20
        reasons.append("Loan large relative to income")

    # credit score
    if credit_score >= 750:
        score += 30
    elif credit_score >= 650:
        score += 10
    else:
        score -= 20
        reasons.append("Low credit score")

    # employment
    if employment_years >= 5:
        score += 15
    elif employment_years >= 2:
        score += 5
    else:
        score -= 10
        reasons.append("Short employment history")

    # existing debt burden
    debt_ratio = existing_debt / max(income*12, 1)
    if debt_ratio < 0.2:
        score += 10
    elif debt_ratio < 0.5:
        score += 0
    else:
        score -= 10
        reasons.append("High existing debt")

    # model score (if available)
    model = load_loan_model() if use_model else None
    model_prob = None
    if model is not None:
        # create dataframe with same feature names as prepare_features_for_loan expected
        X = pd.DataFrame({
            "loan_amount": [loan_amount],
            "income": [income],
            "credit_score": [credit_score],
            "age": [age],
            "existing_debt": [existing_debt],
            "term": [application.get("term", 12)],
            "employment_years": [employment_years]
        })
        try:
            model_prob = float(model.predict_proba(X.fillna(0))[:, 1][0])
            score += model_prob * 100 * 0.2  # weight model modestly
        except Exception:
            model_prob = None

    # final decision thresholds
    decision = "review"
    if score >= 60:
        decision = "approve"
    elif score < 30:
        decision = "deny"

    return {
        "score": round(float(score), 2),
        "decision": decision,
        "reasons": reasons,
        "model_prob": model_prob
    }
