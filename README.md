# churn-prediction-backend

## Overview
This repository contains the backend service for a customer churn prediction system.
It exposes a machine learning model via a REST API to perform batch churn risk prediction on customer data.

The service is designed to support real-world usage patterns such as CSV-based batch inference
and integration with a frontend dashboard.

---

## Problem Statement
Customer churn directly impacts revenue in subscription-based businesses.
The goal of this service is to identify customers with a high likelihood of churn
so that retention actions can be taken proactively.

---

## Approach
- Trained a supervised classification model on customer subscription data
- Used logistic regression for interpretability and probability-based risk scoring
- Performed feature preprocessing and handling of missing values
- Tuned decision thresholds to prioritize recall for high-risk customers
- Exposed predictions through a FastAPI-based REST interface

---

## API Endpoints

### POST /predict/batch
Accepts a CSV file containing customer records and returns churn risk predictions.

Request:
- Content-Type: multipart/form-data
- Field: file (CSV)

Response:
{
  "total_customers": 100,
  "churn_risk_count": 28,
  "results": [
    {
      "tenure": 12,
      "MonthlyCharges": 79.4,
      "Contract": "Month-to-month",
      "PaymentMethod": "Electronic check",
      "churn_probability": 0.38,
      "churn_prediction": 1
    }
  ]
}

---

## Tech Stack
- Python
- FastAPI
- scikit-learn
- pandas
- NumPy

---

## Running Locally

pip install -r requirements.txt

python -m uvicorn app.main:app --reload

API available at:
http://localhost:8000

Swagger UI:
http://localhost:8000/docs

---

## Key Learnings
- Designing ML systems around business metrics rather than accuracy alone
- Importance of probability calibration and threshold selection
- Structuring ML code for reuse in production APIs
- Handling real-world integration issues such as file uploads and CORS

---

## Notes
- The frontend dashboard consuming this API is maintained in a separate repository
- Model artifacts are loaded at runtime to keep inference lightweight
