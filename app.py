from fastapi import FastAPI, UploadFile, File
import io
import joblib
import json
from pydantic import BaseModel
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str


RAW_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod"]


app = FastAPI(title="Churn Prediction API")


model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/model_metadata.json") as f:
    metadata = json.load(f)


FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]

feature_importance = dict(zip(FEATURES, model.coef_[0]))


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(customer: CustomerInput):

    df = pd.DataFrame([customer.dict()])

    # one hot encode
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

    # 1. keep only raw required columns
    df_raw = df[RAW_COLUMNS].copy()

    # 2. feature engineering (same as training)
    df_raw["avg_monthly_spend"] = df_raw["MonthlyCharges"]
    df_raw["total_spend_estimate"] = df_raw["MonthlyCharges"] * df_raw["tenure"]

    # 3. one-hot encode
    df_encoded = pd.get_dummies(df_raw)

    # 4. align with training features
    df_encoded = df_encoded.reindex(columns=FEATURES, fill_value=0)

    # 5. scale
    df_scaled = scaler.transform(df_encoded)

    # df_scaled = scaler.transform(df_model)

    probs = model.predict_proba(df_scaled)[:, 1]

    predictions = (probs >= THRESHOLD).astype(int)

    df["churn_probability"] = probs.round(3)
    df["churn_prediction"] = predictions

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

    # sort by absolute impact
    sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feature, impact in sorted_impacts[:top_k]:
        if impact > 0:
            reasons.append(f"{feature} increases churn risk")
        else:
            reasons.append(f"{feature} reduces churn risk")

    return reasons
