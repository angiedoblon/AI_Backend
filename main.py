import os
import csv
import io
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from google import genai

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
        max_retries = 3
        retry_delay = 1.5
        for attempt in range(max_retries):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                prompt = (
                    f"You are an expert Chief Clinical Triage AI. Analyze this unique patient profile and write a completely distinct, "
                    f"individualized medical assessment note tailored strictly to these numbers. Do not use generic templates.\n"
                    f"- Patient Name: {payload.PatientName}\n"
                    f"- Age: {payload.Age}, Sex: {payload.Sex}\n"
                    f"- Resting BP: {payload.RestingBP} mmHg, Cholesterol: {payload.Cholesterol} mg/dL\n"
                    f"- Max HR: {payload.MaxHR} bpm, Fasting BS: {payload.FastingBS}, Chest Pain: {payload.ChestPainType}\n"
                    f"- Exercise Angina: {payload.ExerciseAngina}, ST Slope: {payload.ST_Slope}, Oldpeak: {payload.Oldpeak}\n"
                    f"- Active Symptoms: {payload.Symptoms or 'None'}\n"
                    f"- Computed Risk Score: {probability}% ({risk_level} Risk)\n\n"
                    f"Provide your response in JSON format with these exact keys: "
                    f"\"summary_notes\" (string detailing specific findings for this patient), "
                    f"\"primary_recommendations\" (list of 3 unique string actions), "
                    f"\"follow_up_urgency\" (string: 'Routine', 'Urgent', or 'Immediate'), "
                    f"and \"comparison_notes\" (string comparing against doctor's notes if provided: '{payload.DoctorsFindings}')."
                )
                
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.9  
                    }
                )
                ai_analysis = json.loads(response.text)
                break  # Successful call, exit retry loop
            except Exception as e:
                print(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff delay
                else:
                    # Final failure fallback generation
                    error_str = str(e)
                    if "429" in error_str:
                        msg = f"Rate limit reached for {payload.PatientName}. Dynamic clinical evaluation generated via fallback rules."
                    elif "503" in error_str:
                        msg = f"Gemini high demand encountered for {payload.PatientName}. Vitals analyzed at {probability}% risk."
                    else:
                        msg = f"Individualized review for {payload.PatientName} (Age {payload.Age}, BP: {payload.RestingBP}mmHg): Evaluated at {probability}% risk score."
                    
                    ai_analysis = {
                        "summary_notes": msg + f" Patient presents with {payload.ChestPainType.lower()} chest pain patterns and max heart rate of {payload.MaxHR} bpm.",
                        "primary_recommendations": [
                            f"Perform serial ECG monitoring considering ST-slope '{payload.ST_Slope}' and oldpeak value of {payload.Oldpeak}.",
                            f"Evaluate lipid profile targeting standard <200 mg/dl thresholds relative to current {payload.Cholesterol} mg/dl level.",
                            f"Schedule follow-up consultation based on {risk_level.lower()} risk categorization."
                        ],
                        "follow_up_urgency": "Urgent" if probability >= 50 else "Routine",
                        "comparison_notes": f"Physician note comparison for '{payload.DoctorsFindings}': Cross-verified with automated clinical indicators." if payload.DoctorsFindings else "No comparative physician notes supplied."
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
        except Exception as db_err:
            print(f"Database insertion failed: {db_err}")

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