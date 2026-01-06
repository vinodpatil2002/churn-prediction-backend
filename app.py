from fastapi import FastAPI, UploadFile, File
import io
import os
import joblib
import json
from pydantic import BaseModel
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(title="Churn Prediction API")

model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]

feature_importance = dict(zip(FEATURES, model.coef_[0]))


class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str


RAW_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://churn-prediction-frontend-theta.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    df = pd.DataFrame([customer.dict()])
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=FEATURES, fill_value=0)

    df_scaled = scaler.transform(df_encoded)
    prob = model.predict_proba(df_scaled)[0][1]
    prediction = int(prob >= THRESHOLD)

    reasons = explain_prediction(df_encoded)

    return {
        "churn_probability": round(float(prob), 3),
        "churn_prediction": prediction,
        "reasons": reasons,
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    COLUMN_RENAME_MAP = {
        "Tenure": "tenure",
        "Monthly Charges": "MonthlyCharges",
        "Payment Method": "PaymentMethod",
    }
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return {
            "error": "Invalid input file",
            "missing_columns": missing,
            "received_columns": list(df.columns),
        }

    df_raw = df[RAW_COLUMNS].copy()

    df_encoded = pd.get_dummies(df_raw)
    df_encoded = df_encoded.reindex(columns=FEATURES, fill_value=0)

    df_scaled = scaler.transform(df_encoded)
    probs = model.predict_proba(df_scaled)[:, 1]
    predictions = (probs >= THRESHOLD).astype(int)

    df["churn_probability"] = probs.round(3)
    df["churn_prediction"] = np.where(predictions == 1, "High Risk", "Low Risk")

    return {
        "total_customers": len(df),
        "churn_risk_count": int(predictions.sum()),
        "results": df.to_dict(orient="records"),
    }


def explain_prediction(df_encoded_row, top_k=3):
    impacts = {}

    for col in df_encoded_row.columns:
        value = df_encoded_row[col].values[0]
        coef = feature_importance.get(col, 0)
        impacts[col] = value * coef

    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feature, impact in sorted_impacts[:top_k]:
        if impact > 0:
            reasons.append(f"{feature} increases churn risk")
        else:
            reasons.append(f"{feature} reduces churn risk")

    return reasons
