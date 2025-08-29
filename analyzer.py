import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def load_transactions(path: str) -> pd.DataFrame:
    """Load a CSV of transactions. Attempts to coerce common columns."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    amount_col = None
    for cand in ["amount", "transaction_amount", "amt", "purchase_amount"]:
        if cand in cols:
            amount_col = cols[cand]
            break
    if amount_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            amount_col = num_cols[0]
    for cand in ["date", "transaction_date", "time", "timestamp"]:
        if cand in cols:
            try:
                df[cols[cand]] = pd.to_datetime(df[cols[cand]])
            except Exception:
                pass
    return df

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02, random_state: int = 42):
    """Run IsolationForest on numeric features to flag anomalous transactions."""
    if df.empty:
        return df.assign(anomaly=False, anomaly_score=np.nan)
    num_df = df.select_dtypes(include=["number"]).copy()
    if num_df.shape[1] == 0:
        str_df = df.astype(str).applymap(len)
        num_df = str_df
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    model = IsolationForest(
        n_estimators=200,
        contamination=min(max(contamination, 0.001), 0.25),
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(num_df.values)
    scores = model.decision_function(num_df.values)
    preds = model.predict(num_df.values)  # -1 anomalous, 1 normal
    out = df.copy()
    out["anomaly_score"] = scores
    out["anomaly"] = preds == -1
    return out

def kpis(df: pd.DataFrame):
    """Compute simple KPIs from a transactions-like table."""
    out = {}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        amount_col = num_cols[0]
        out["total_volume"] = float(df[amount_col].sum())
        out["avg_txn"] = float(df[amount_col].mean())
        out["txn_count"] = int(df.shape[0])
    else:
        out["txn_count"] = int(df.shape[0])
    return out
