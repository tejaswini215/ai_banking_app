import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_PATH = "loan_model.pkl"

# ------------------ Transaction Utilities ------------------

def load_transactions(path: str) -> pd.DataFrame:
    """Load transactions CSV."""
    return pd.read_csv(path)

def kpis(df: pd.DataFrame) -> dict:
    """Compute KPIs for dashboard."""
    if df.empty:
        return {"txn_count": 0, "total_volume": 0, "avg_txn": 0}
    return {
        "txn_count": len(df),
        "total_volume": df["Amount"].sum() if "Amount" in df else 0,
        "avg_txn": df["Amount"].mean() if "Amount" in df else 0,
    }

# analyzer.py
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, contamination=0.02):
    """Detect anomalies in transactions using Isolation Forest."""
    if df.empty:
        return df

    # Assume transaction amount column
    amount_col = None
    for col in df.columns:
        if "amount" in col.lower():
            amount_col = col
            break

    if not amount_col:
        raise ValueError("No 'amount' column found in dataset")

    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(df[[amount_col]])
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})  # 1=anomaly
    return df


# ------------------ Loan Assessment Utilities ------------------

def prepare_features_for_loan(df: pd.DataFrame):
    """Prepare features (X) and labels (y) for loan model."""
    if "Loan_Status" not in df.columns:
        raise ValueError("Dataset must contain 'Loan_Status' column.")

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"].apply(lambda x: 1 if str(x).lower() in ["y", "yes", "approved", "1"] else 0)
    return X, y

def train_loan_model(df: pd.DataFrame):
    """Train and save a RandomForest loan approval model."""
    X, y = prepare_features_for_loan(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)

    return model, acc

def load_loan_model():
    """Load trained loan model if available."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def assess_loan(applicant_data: dict) -> str:
    """Assess loan application using trained model."""
    model = load_loan_model()
    if model is None:
        return "⚠️ Loan model not trained yet."

    df = pd.DataFrame([applicant_data])
    pred = model.predict(df)[0]
    return "✅ Loan Approved" if pred == 1 else "❌ Loan Rejected"
