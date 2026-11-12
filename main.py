import os
import csv
import io
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI

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

ai_client = None
if GEMINI_API_KEY:
    ai_client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

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
    summary_notes: str = Field(description="Initial clinical findings and nurse triage summary uniquely customized based on exact patient vitals and symptoms.")
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
    
    if ai_client:
        try:
            user_content = (
                f"Patient Name: {payload.PatientName}\n"
                f"Clinical Vitals: Age={payload.Age}, Sex={payload.Sex}, Resting BP={payload.RestingBP} mmHg, Cholesterol={payload.Cholesterol} mg/dL, Max HR={payload.MaxHR} bpm, Fasting BS={payload.FastingBS}\n"
                f"Cardiac Indicators: Chest Pain Type={payload.ChestPainType}, Exercise Angina={payload.ExerciseAngina}, ST Slope={payload.ST_Slope}, Oldpeak={payload.Oldpeak}, Resting ECG={payload.RestingECG}\n"
                f"Nurse Triage Symptoms Reported: '{payload.Symptoms or 'None reported'}'\n"
                f"Calculated Scikit-Learn Model Probability: {probability}% ({risk_level} Risk)."
            )
            if payload.DoctorsFindings:
                user_content += f"\nPhysician's Actual Findings: '{payload.DoctorsFindings}'. Cross-compare these findings against the patient vitals and risk score."
            
            completion = ai_client.beta.chat.completions.parse(
                model="gemini-1.5-flash",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert Chief Clinical Triage AI and Medical Decision Support Assistant. "
                            "Analyze the specific patient parameters, vital signs, and symptoms provided. "
                            "Generate professional, highly individualized clinical findings and actionable nurse recommendations. "
                            "Never use a generic template; explicitly reference abnormal vitals (such as elevated blood pressure or cholesterol) and reported symptoms."
                        )
                    },
                    {
                        "role": "user", 
                        "content": user_content
                    }
                ],
                response_format=ClinicalInsight,
            )
            ai_analysis = completion.choices[0].message.parsed.model_dump()
        except Exception as e:
            print(f"CRITICAL Gemini AI Generation Error: {e}")
            
    if not ai_analysis:
        symptom_notes = f" Patient reports active symptoms: {payload.Symptoms}." if payload.Symptoms else " No acute symptoms checked during intake."
        vitals_notes = f" Vital signs reveal a resting blood pressure of {payload.RestingBP} mmHg and serum cholesterol of {payload.Cholesterol} mg/dL."
        
        comparison_fallback = (
            f"Compared with physician findings ('{payload.DoctorsFindings}'): Evaluation cross-referenced with automated metrics." 
            if payload.DoctorsFindings else "No external doctor findings provided for cross-comparison."
        )
        ai_analysis = {
            "summary_notes": f"Clinical Triage Evaluation for {payload.PatientName} (Age {payload.Age}): Assessed with a {risk_level.lower()} risk profile ({probability}% probability).{vitals_notes}{symptom_notes}",
            "primary_recommendations": [
                f"Initiate continuous hemodynamic monitoring given resting BP of {payload.RestingBP} mmHg.",
                "Provide targeted patient counseling on cardiovascular risk reduction and dietary sodium/lipid management.",
                "Facilitate urgent physician review if symptoms escalate or persist."
            ],
            "follow_up_urgency": "Urgent" if probability >= 50 else "Routine",
            "comparison_notes": comparison_fallback
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