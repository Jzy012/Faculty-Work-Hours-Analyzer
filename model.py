# model.py
import pandas as pd
from sklearn.ensemble import IsolationForest

class AttendanceModel:
    def __init__(self, historical_path="data/historical.csv"):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.historical_path = historical_path
        self.train_model()

    def preprocess(self, df):
        df['time_in'] = pd.to_datetime(df['time_in'].astype(str))
        df['time_out'] = pd.to_datetime(df['time_out'].astype(str))
        df['work_hours'] = (df['time_out'] - df['time_in']).dt.total_seconds() / 3600
        return df

    def train_model(self):
        df = pd.read_csv(self.historical_path)
        df = self.preprocess(df)
        self.model.fit(df[['work_hours']])
        print("✅ Model trained on historical data")

    def predict(self, new_df):
        new_df = self.preprocess(new_df)
        new_df['anomaly'] = self.model.predict(new_df[['work_hours']])
        new_df['status'] = new_df['anomaly'].apply(lambda x: 'Unusual' if x == -1 else 'Normal')
        return new_df
