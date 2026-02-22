from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# -------------------------
# Load Model Properly
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "features.pkl"))

# -------------------------
# Create FastAPI App
# -------------------------

app = FastAPI(title="Fraud Detection API")

# -------------------------
# Request Schema
# -------------------------

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# -------------------------
# Home Endpoint
# -------------------------

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

# -------------------------
# Predict Endpoint
# -------------------------

@app.post("/predict")
def predict(txn: Transaction):

    input_df = pd.DataFrame([txn.dict()])

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    probability = model.predict_proba(input_df)[0][1]

    # Financial logic
    FRAUD_LOSS = 1000
    FALSE_BLOCK_COST = 10

    expected_loss_allow = probability * FRAUD_LOSS
    expected_loss_block = (1 - probability) * FALSE_BLOCK_COST

    if expected_loss_allow > expected_loss_block:
        decision = "Block Transaction 🚨"
    else:
        decision = "Allow Transaction ✅"

    return {
        "fraud_probability": round(float(probability), 4),
        "expected_loss_if_allowed": round(float(expected_loss_allow), 2),
        "expected_loss_if_blocked": round(float(expected_loss_block), 2),
        "recommended_action": decision
    }