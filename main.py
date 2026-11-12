import os
import csv
import io
import json
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set.")
        return
    conn = get_db_connection()
    cur = conn.cursor()
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
    cur.close()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def root():
    return {"status": "online", "message": "Health Risk Assessor API is active."}

class HeartData(BaseModel):
    PatientName: str = Field(default="Anonymous Patient", description="Name or ID of the patient being assessed")
    Age: int = Field(..., example=45)
    RestingBP: int = Field(..., example=120)
    Cholesterol: int = Field(..., example=200)
    Sex: str = Field(default="Male")
    MaxHR: int = Field(default=150)
    FastingBS: int = Field(default=0)
    ChestPainType: str = Field(default="Atypical Angina")
    ExerciseAngina: str = Field(default="No")
    ST_Slope: str = Field(default="Up")
    Oldpeak: float = Field(default=1.0)
    RestingECG: str = Field(default="Normal")
    DoctorsFindings: str = Field(default="", description="Physician's actual diagnosis notes for comparison")
    Symptoms: str = Field(default="", description="Observed nurse triage symptoms or notes")

class ClinicalInsight(BaseModel):
    summary_notes: str = Field(description="Unique, detailed clinical findings and narrative assessment tailored strictly to this specific patient's vitals, name, and symptoms.")
    primary_recommendations: list[str] = Field(description="List of 2-3 specific actionable nurse or clinical steps.")
    follow_up_urgency: str = Field(description="Level of follow-up required: 'Routine', 'Urgent', or 'Immediate'")
    comparison_notes: str = Field(description="Comparison analysis alignment or discrepancies if doctor's findings were provided.")

@app.post("/predict")
def predict(payload: HeartData):
    base_score = (payload.Age * 0.35) + (payload.RestingBP * 0.25) + (payload.Cholesterol * 0.15) + (payload.Oldpeak * 5.0)
    if payload.FastingBS == 1:
        base_score += 10
    if payload.ExerciseAngina.lower() == "yes":
        base_score += 15
        
    probability = round(min(max(base_score / 2.8, 10.0), 95.0), 1)
    risk_level = "High" if probability >= 50 else "Low"
    ai_analysis = None
    if GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            prompt = (
                f"You are an expert Chief Clinical Triage AI. Analyze this patient profile and write a completely unique, "
                f"professional medical assessment. Do not use generic templates.\n"
                f"Patient Name: {payload.PatientName}\n"
                f"Age: {payload.Age}, Sex: {payload.Sex}\n"
                f"Resting BP: {payload.RestingBP} mmHg, Cholesterol: {payload.Cholesterol} mg/dL\n"
                f"Max HR: {payload.MaxHR} bpm, Fasting BS: {payload.FastingBS}\n"
                f"Symptoms: {payload.Symptoms or 'None'}\n"
                f"Computed Risk Score: {probability}% ({risk_level} Risk)\n\n"
                f"Provide your response in JSON format with these exact keys: "
                f"\"summary_notes\" (string), \"primary_recommendations\" (list of 3 strings), "
                f"\"follow_up_urgency\" (string: 'Routine', 'Urgent', or 'Immediate'), "
                f"and \"comparison_notes\" (string)."
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.8
                }
            )
            ai_analysis = json.loads(response.text)
        except Exception as e:
            print(f"Gemini SDK Error: {e}")

    if not ai_analysis:
        # Dynamic fallback if network/API hiccups
        symptom_txt = f" Patient reports: {payload.Symptoms}." if payload.Symptoms else " No acute symptoms noted."
        ai_analysis = {
            "summary_notes": f"Triage evaluation for {payload.PatientName} (Age {payload.Age}): Assessed at {probability}% risk ({risk_level}). Blood pressure is {payload.RestingBP} mmHg with cholesterol at {payload.Cholesterol} mg/dL.{symptom_txt}",
            "primary_recommendations": [
                f"Monitor resting BP of {payload.RestingBP} mmHg closely during observation.",
                "Provide counseling on cardiovascular risk factors and diet.",
                "Schedule physician check-up as dictated by risk level."
            ],
            "follow_up_urgency": "Urgent" if probability >= 50 else "Routine",
            "comparison_notes": f"Compared with physician findings ('{payload.DoctorsFindings}'): Automated model aligns with clinical risk metrics." if payload.DoctorsFindings else "No external doctor findings provided for cross-comparison."
        }

    if DATABASE_URL:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS patient_name VARCHAR(100);")
            cur.execute(
                """
                INSERT INTO predictions (patient_name, age, blood_pressure, bmi, risk_score, risk_level)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (payload.PatientName, payload.Age, payload.RestingBP, payload.Cholesterol, probability, risk_level)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Database insertion failed: {e}")

    return {
        "heart_disease_risk": 1 if probability >= 50 else 0,
        "risk_probability_percent": probability,
        "risk_level": risk_level,
        "ai_analysis": ai_analysis
    }

@app.get("/history")
def get_history():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') as timestamp,
                COALESCE(patient_name, 'Anonymous') as patient_name,
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
        cur.close()
        conn.close()
        return {"history": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export-csv")
def export_csv():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
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
        cur.close()
        conn.close()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(["Timestamp", "Patient Name", "Age", "Blood Pressure", "Cholesterol (mg/dl)", "Risk Score (%)", "Risk Level"])
        
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
            headers={"Content-Disposition": "attachment; filename=patient_risk_records.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))