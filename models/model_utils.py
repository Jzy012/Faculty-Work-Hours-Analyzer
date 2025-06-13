import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "anomaly_model.pkl")
HISTORICAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "historical.csv")

# --- Feature engineering helpers --- #
def extract_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday

    # Parse times as datetime.time objects, handling possible errors
    def parse_time(t):
        try:
            return datetime.strptime(t, "%H:%M").time()
        except Exception:
            return np.nan

    df["time_in_dt"] = df["time_in"].apply(parse_time)
    df["time_out_dt"] = df["time_out"].apply(parse_time)

    # Calculate work_hours row by row (handle missing/invalid times)
    def calc_work_hours(row):
        if pd.isnull(row["time_in_dt"]) or pd.isnull(row["time_out_dt"]):
            return np.nan
        time_in = datetime.combine(datetime.today(), row["time_in_dt"])
        time_out = datetime.combine(datetime.today(), row["time_out_dt"])
        delta = (time_out - time_in).total_seconds() / 3600
        # Handle negative (overnight) or weird cases
        return delta if delta >= 0 else np.nan

    df["work_hours"] = df.apply(calc_work_hours, axis=1)
    df["late_in"] = df["time_in_dt"].apply(lambda t: int(t.hour * 60 + t.minute > 550) if pd.notnull(t) else 0)
    df["early_out"] = df["time_out_dt"].apply(lambda t: int(t.hour * 60 + t.minute < 960) if pd.notnull(t) else 0)

    # Features for model
    feature_cols = ["weekday", "work_hours", "late_in", "early_out"]
    return df, df[feature_cols].fillna(-1)

def train_model_from_historical():
    df = pd.read_csv(HISTORICAL_CSV)
    df, X = extract_features(df)
    model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model_from_historical()
    return joblib.load(MODEL_PATH)

def predict_anomalies(uploaded_df):
    model = load_model()
    df, X = extract_features(uploaded_df)
    preds = model.predict(X)
    df["anomaly"] = np.where(preds == -1, "Unusual", "Normal")
    return df