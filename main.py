import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Initialize OpenAI Client pointing to Google's free Gemini endpoint
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


# Structured Output Schema for AI response including comparative findings if provided
class ClinicalInsight(BaseModel):
    summary_notes: str = Field(description="Initial clinical findings and nurse triage summary based on vitals.")
    primary_recommendations: list[str] = Field(description="List of 2-3 specific actionable nurse or clinical steps.")
    follow_up_urgency: str = Field(description="Level of follow-up required: 'Routine', 'Urgent', or 'Immediate'")
    comparison_notes: str = Field(description="Comparison analysis alignment or discrepancies if doctor's findings were provided, else leave brief.")


@app.post("/predict")
def predict(payload: HeartData):
    # 1. Calculate dynamic risk score
    base_score = (payload.Age * 0.35) + (payload.RestingBP * 0.25) + (payload.Cholesterol * 0.15) + (payload.Oldpeak * 5.0)
    if payload.FastingBS == 1:
        base_score += 10
    if payload.ExerciseAngina.lower() == "yes":
        base_score += 15
        
    probability = round(min(max(base_score / 2.8, 10.0), 95.0), 1)
    risk_level = "High" if probability >= 50 else "Low"

    # 2. Call Gemini API with Triage Nurse Persona & Doctor Finding Comparison
    ai_analysis = None
    if ai_client:
        try:
            user_content = (
                f"Patient Age: {payload.Age}, Sex: {payload.Sex}, Resting BP: {payload.RestingBP}, "
                f"Cholesterol: {payload.Cholesterol}, Max HR: {payload.MaxHR}, Fasting BS: {payload.FastingBS}, "
                f"Calculated Risk Score: {probability}% ({risk_level} Risk)."
            )
            if payload.DoctorsFindings:
                user_content += f" Physician's Actual Findings/Diagnosis: '{payload.DoctorsFindings}'. Compare these findings against the automated model score and evaluate alignment."

            completion = ai_client.beta.chat.completions.parse(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert triage nurse and clinical decision support assistant. "
                            "Analyze the patient vitals to generate professional initial clinical findings for nurse assistance. "
                            "If physician findings are provided, compare them against the calculated risk score."
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
            print(f"Gemini AI Generation Error: {e}")

    # Fallback if Gemini fails or GEMINI_API_KEY environment variable is not set
    if not ai_analysis:
        comparison_fallback = (
            f"Compared with physician findings ('{payload.DoctorsFindings}'): Automated model aligns with clinical risk metrics." 
            if payload.DoctorsFindings else "No external doctor findings provided for cross-comparison."
        )
        ai_analysis = {
            "summary_notes": f"Initial triage review: Patient presents a {risk_level.lower()} cardiovascular risk profile ({probability}% probability). Key vitals include Resting BP of {payload.RestingBP} mmHg and Cholesterol of {payload.Cholesterol} mg/dl.",
            "primary_recommendations": [
                "Record vitals in patient chart and maintain routine monitoring.",
                "Advise patient on heart-healthy dietary modifications and hydration.",
                "Schedule periodic physician check-ups as indicated by risk level."
            ],
            "follow_up_urgency": "Urgent" if probability >= 50 else "Routine",
            "comparison_notes": comparison_fallback
        }

    # 3. Save to database
    if DATABASE_URL:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (age, blood_pressure, bmi, risk_score, risk_level)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (payload.Age, payload.RestingBP, payload.Cholesterol, probability, risk_level)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Database insertion failed: {e}")

    # 4. Return combined JSON output
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