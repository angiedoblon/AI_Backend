
import os
from dotenv import load_dotenv
import csv
import io
import json
import time
import sys
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from google import genai

# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

app = FastAPI(title="Health Risk Assessor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "healthrisk_model"
SUPPORT_DIR = MODEL_DIR / "healthriskfm_support_modules"

PIPELINE_PATH = MODEL_DIR / "pipeline.joblib"
MODEL_PATH = MODEL_DIR / "model.joblib"

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

# The custom classes used by the saved preprocessing pipeline
# must be importable before joblib loads the saved objects.
sys.path.insert(0, str(SUPPORT_DIR))

pipeline = None
heart_model = None

# =========================================================
# 2. LOAD TRAINED HEALTHRISKFM MODEL
# =========================================================

def load_healthrisk_model():
    global pipeline, heart_model

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {PIPELINE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading HealthRiskFM model...")

    pipeline = joblib.load(PIPELINE_PATH)
    heart_model = joblib.load(MODEL_PATH)

    if not hasattr(heart_model, "predict_proba"):
        raise RuntimeError(
            "The loaded model does not support predict_proba()."
        )

    if not hasattr(heart_model, "feature_names_in_"):
        raise RuntimeError(
            "The model does not contain its expected feature names."
        )

    print("HealthRiskFM loaded successfully.")
    print("Expected features:", len(heart_model.feature_names_in_))

# =========================================================
# 3. DATABASE CONNECTION
# =========================================================

def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )

def init_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set. Database history disabled.")
        return

    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    patient_name VARCHAR(100),
                    age INT,
                    blood_pressure INT,
                    bmi INT,
                    risk_score FLOAT,
                    risk_level VARCHAR(20)
                );
            """)

        conn.commit()
        print("Database initialized successfully.")

    finally:
        conn.close()

# =========================================================
# 4. STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():
    load_healthrisk_model()
    init_db()

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Health Risk Assessor API is active.",
        "model": "HealthRiskFM Random Forest"
    }

# =========================================================
# 5. PATIENT INPUT MODEL
# =========================================================

class HeartData(BaseModel):

    PatientName: str = "Anonymous Patient"

    Age: int = Field(..., ge=1, le=120)
    RestingBP: int = Field(..., ge=40, le=240)
    Cholesterol: int = Field(..., ge=50, le=600)

    Sex: str = "M"
    MaxHR: int = Field(default=150, ge=60, le=220)
    FastingBS: int = Field(default=0, ge=0, le=1)

    ChestPainType: str = "ATA"
    ExerciseAngina: str = "N"
    ST_Slope: str = "Up"

    Oldpeak: float = Field(default=1.0, ge=-2.5, le=6.2)
    RestingECG: str = "Normal"

    DoctorsFindings: str = ""
    Symptoms: str = ""

# =========================================================
# 6. HEALTHRISKFM PREDICTION
# =========================================================

def predict_with_healthriskfm(payload: HeartData):

    if pipeline is None or heart_model is None:
        raise RuntimeError("HealthRiskFM model is not loaded.")

    patient_data = pd.DataFrame([{
        "Age": payload.Age,
        "Sex": payload.Sex,
        "ChestPainType": payload.ChestPainType,
        "RestingBP": payload.RestingBP,
        "Cholesterol": payload.Cholesterol,
        "FastingBS": payload.FastingBS,
        "RestingECG": payload.RestingECG,
        "MaxHR": payload.MaxHR,
        "ExerciseAngina": payload.ExerciseAngina,
        "Oldpeak": payload.Oldpeak,
        "ST_Slope": payload.ST_Slope
    }])

    # Apply the original fitted preprocessing pipeline.
    processed_data = pipeline.transform(patient_data)

    # Match the exact feature order used during model training.
    expected_features = list(heart_model.feature_names_in_)

    missing_features = [
        feature for feature in expected_features
        if feature not in processed_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    processed_data = processed_data.reindex(
        columns=expected_features
    )

    # Generate the trained Random Forest prediction.
    prediction = int(heart_model.predict(processed_data)[0])

    probabilities = heart_model.predict_proba(processed_data)[0]

    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(
            heart_model.classes_,
            probabilities
        )
    }

    if 1 not in class_probabilities or 0 not in class_probabilities:
        raise ValueError("Expected model classes 0 and 1.")

    # Dataset class 1 represents heart disease.
    disease_probability = round(
        class_probabilities[1] * 100,
        2
    )

    no_disease_probability = round(
        class_probabilities[0] * 100,
        2
    )

    risk_level = "High" if prediction == 1 else "Low"

    return {
        "prediction": prediction,
        "disease_probability": disease_probability,
        "no_disease_probability": no_disease_probability,
        "risk_level": risk_level
    }

# =========================================================
# 7. PREDICTION API
# =========================================================

@app.post("/predict")
def predict(payload: HeartData):

    try:
        result = predict_with_healthriskfm(payload)

    except Exception as model_error:
        print(f"HealthRiskFM prediction failed: {model_error}")
        raise HTTPException(
            status_code=500,
            detail="The AI model could not process this assessment."
        )

    prediction = result["prediction"]
    probability = result["disease_probability"]
    risk_level = result["risk_level"]

    ai_analysis = None


    # =====================================================
    # 8. GEMINI EXPLANATION
    # =====================================================

    if GEMINI_API_KEY:

        try:
            client = genai.Client(api_key=GEMINI_API_KEY)

            prompt = f"""
You are assisting with explanations for an experimental
heart disease risk assessment system.

The HealthRiskFM Random Forest model produced this result:

Age: {payload.Age}
Sex: {payload.Sex}
Resting Blood Pressure: {payload.RestingBP} mmHg
Cholesterol: {payload.Cholesterol} mg/dL
Maximum Heart Rate: {payload.MaxHR}
Fasting Blood Sugar: {payload.FastingBS}
Chest Pain Type: {payload.ChestPainType}
Exercise Angina: {payload.ExerciseAngina}
Resting ECG: {payload.RestingECG}
ST Slope: {payload.ST_Slope}
Oldpeak: {payload.Oldpeak}

Model prediction class: {prediction}
Model class 1 probability: {probability}%
Model classification: {risk_level}

Symptoms: {payload.Symptoms or "None provided"}
Physician findings: {payload.DoctorsFindings or "None provided"}

Explain the model result in simple, understandable language.

Important:
- This is not a confirmed medical diagnosis.
- The percentage is an uncalibrated model probability,
  not a validated clinical risk estimate.
- Symptoms and physician findings are NOT inputs to the
  Random Forest model and did not affect its prediction.
- Do not imply that symptoms or physician findings
  influenced the model classification.
- Do not invent symptoms or medical findings.
- Do not recommend medication or treatment based only
  on this result.
- Encourage professional medical assessment.
- If severe or emergency symptoms are reported, advise
  immediate emergency medical evaluation.

Return valid JSON with exactly these keys:
"summary_notes": string,
"primary_recommendations": array of 3 strings,
"follow_up_urgency": string,
"comparison_notes": string.

Use Routine, Urgent, or Immediate for follow_up_urgency.
Do not determine urgency from the model percentage alone.
"""

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.5
                }
            )

            ai_analysis = json.loads(response.text)

            required_keys = [
                "summary_notes",
                "primary_recommendations",
                "follow_up_urgency",
                "comparison_notes"
            ]

            if not all(key in ai_analysis for key in required_keys):
                raise ValueError("Gemini response is missing required fields.")

            if not isinstance(ai_analysis["primary_recommendations"], list):
                raise ValueError("Gemini recommendations must be a list.")

            if ai_analysis["follow_up_urgency"] not in [
                "Routine", "Urgent", "Immediate"
            ]:
                ai_analysis["follow_up_urgency"] = "Routine"

            print("Gemini personalized explanation generated successfully.")

        except Exception as e:

            print(f"Gemini explanation failed: {e}")

            ai_analysis = {
                "summary_notes": (
                    f"HealthRiskFM classified this assessment as "
                    f"{risk_level}. Its class 1 probability was "
                    f"{probability}%. This is an experimental model "
                    f"result, not a medical diagnosis. Symptoms and "
                    f"physician findings were not used by the "
                    f"Random Forest prediction."
                ),
                "primary_recommendations": [
                    "Discuss the assessment results with a qualified healthcare professional.",
                    "Review the patient's symptoms and clinical history.",
                    "Consider further evaluation as advised by a healthcare professional."
                ],
                "follow_up_urgency": "Routine",
                "comparison_notes": (
                    "The Random Forest prediction was based on its "
                    "18 model features. Symptoms and physician findings "
                    "were provided separately and did not affect the "
                    "prediction."
                )
            }

    else:

        ai_analysis = {
            "summary_notes": (
                f"HealthRiskFM classified this assessment as "
                f"{risk_level}. Its class 1 probability was "
                f"{probability}%. This is an experimental model "
                f"result, not a medical diagnosis."
            ),
            "primary_recommendations": [
                "Discuss the assessment results with a qualified healthcare professional.",
                "Review the patient's symptoms and clinical history.",
                "Consider further evaluation as advised by a healthcare professional."
            ],
            "follow_up_urgency": "Routine",
            "comparison_notes": (
                "Gemini is unavailable because no API key is configured. "
                "Symptoms and physician findings did not affect the "
                "Random Forest prediction."
            )
        }



    # 8.5. SYMPTOM-BASED SAFETY CHECK
    # This changes follow-up guidance only.
    # It does not change the Random Forest prediction.

    symptom_text = (
        f"{payload.Symptoms or ''} "
        f"{payload.DoctorsFindings or ''}"
    ).lower()

    def contains_symptom(phrases):
        for phrase in phrases:
            start = 0

            while True:
                position = symptom_text.find(phrase, start)

                if position == -1:
                    break

                preceding_text = symptom_text[
                    max(0, position - 30):position
                ].strip()


                negations = (
                    "no",
                    "denies",
                    "denied",
                    "without",
                    "not experiencing",
                    "not having"
                )

                if not any(
                    preceding_text == word
                    or preceding_text.endswith(" " + word)
                    for word in negations
                ):
                    return True


                start = position + len(phrase)

        return False

    emergency_symptoms = [
        "severe chest pain",
        "crushing chest pain",
        "severe chest pressure",
        "persistent chest pain",
        "ongoing chest pain",
        "severe shortness of breath",
        "severe difficulty breathing",
        "difficulty breathing at rest",
        "fainting now",
        "just fainted",
        "loss of consciousness",
        "stroke-like symptoms"
    ]

    concerning_symptoms = [
        "chest discomfort",
        "chest pain",
        "chest tightness",
        "chest pressure",
        "shortness of breath",
        "difficulty breathing",
        "breathlessness",
        "exertional dyspnea"
    ]

    if contains_symptom(emergency_symptoms):

        ai_analysis["follow_up_urgency"] = "Immediate"

        ai_analysis["primary_recommendations"] = [
            "Seek emergency medical evaluation immediately for the reported severe or potentially life-threatening symptoms.",
            "Do not rely on the experimental machine-learning result to decide whether emergency care is needed.",
            "Share the reported symptoms and relevant medical information with emergency healthcare professionals."
        ]

    elif contains_symptom(concerning_symptoms):

        ai_analysis["follow_up_urgency"] = "Urgent"

        ai_analysis["primary_recommendations"] = [
            "Arrange prompt medical evaluation for the reported chest or breathing symptoms.",
            "Discuss the symptoms and assessment measurements with a qualified healthcare professional.",
            "Do not use the experimental model result alone to determine medical urgency or treatment."
        ]

    print(
        "Symptom safety check completed:",
        ai_analysis["follow_up_urgency"]
    )

    # =====================================================
    # 9. SAVE PREDICTION TO POSTGRESQL
    # =====================================================

    if DATABASE_URL:

        conn = None

        try:
            conn = get_db_connection()

            with conn.cursor() as cur:

                cur.execute(
                    """
                    INSERT INTO predictions (
                        patient_name,
                        age,
                        blood_pressure,
                        bmi,
                        risk_score,
                        risk_level
                    )
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        payload.PatientName,
                        payload.Age,
                        payload.RestingBP,
                        payload.Cholesterol,
                        probability,
                        risk_level
                    )
                )

            conn.commit()

        except Exception as db_error:

            if conn:
                conn.rollback()

            print(f"Database insertion failed: {db_error}")

        finally:
            if conn:
                conn.close()

    # =====================================================
    # 10. API RESPONSE
    # =====================================================

    return {
        "heart_disease_risk": prediction,
        "risk_probability_percent": probability,
        "risk_score": probability,
        "risk_level": risk_level,
        "prediction_class": prediction,
        "class_0_probability_percent": result["no_disease_probability"],
        "class_1_probability_percent": probability,
        "model_name": "HealthRiskFM Random Forest",
        "ai_analysis": ai_analysis,
        "disclaimer": (
            "Experimental machine-learning output. "
            "Not a medical diagnosis or calibrated clinical risk estimate."
        )
    }

# =========================================================
# 11. PREDICTION HISTORY
# =========================================================

@app.get("/history")
def get_history():

    if not DATABASE_URL:
        return {"history": []}

    conn = None

    try:
        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') AS timestamp,
                    COALESCE(patient_name, 'Anonymous') AS patient_name,
                    age,
                    blood_pressure,
                    bmi,
                    risk_score,
                    risk_level
                FROM predictions
                ORDER BY id DESC
                LIMIT 50;
            """)

            records = cur.fetchall()

        return {"history": records}

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve prediction history."
        )

    finally:
        if conn:
            conn.close()

# =========================================================
# 12. EXPORT PREDICTION HISTORY TO CSV
# =========================================================

@app.get("/export-csv")
def export_csv():

    if not DATABASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured."
        )

    conn = None

    try:
        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    timestamp,
                    patient_name,
                    age,
                    blood_pressure,
                    bmi,
                    risk_score,
                    risk_level
                FROM predictions
                ORDER BY id DESC;
            """)

            records = cur.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Timestamp",
            "Patient Name",
            "Age",
            "Blood Pressure",
            "Cholesterol (mg/dl)",
            "Heart Disease Probability (%)",
            "Risk Level"
        ])

        for row in records:

            writer.writerow([
                row["timestamp"],
                row["patient_name"],
                row["age"],
                row["blood_pressure"],
                row["bmi"],
                row["risk_score"],
                row["risk_level"]
            ])

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                "attachment; filename=patient_risk_records.csv"
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail="Unable to export prediction history."
        )

    finally:
        if conn:
            conn.close()
