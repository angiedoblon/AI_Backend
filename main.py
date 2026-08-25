from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI(title="Health Risk Prediction API")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Random Forest model
model = joblib.load("model.joblib")

# Retrieve exact feature names expected by the trained model
EXPECTED_FEATURES = model.feature_names_in_

@app.get("/")
def home():
    return {"status": "AI_Backend is live and running!"}

@app.post("/predict")
def predict_risk(data: dict):
    # 1. Convert incoming JSON request into a DataFrame
    raw_df = pd.DataFrame([data])
    
    # 2. Convert categorical text columns to one-hot encoded columns
    encoded_df = pd.get_dummies(raw_df)
    
    # 3. Align columns to match the exact model training schema (fills missing features with 0)
    aligned_df = encoded_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
    
    # 4. Perform prediction
    prediction = model.predict(aligned_df)[0]
    
    risk_percentage = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(aligned_df)[0][1]
        risk_percentage = round(float(probability) * 100, 2)

    return {
        "heart_disease_risk": int(prediction),
        "risk_probability_percent": risk_percentage
    }