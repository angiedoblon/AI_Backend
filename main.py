import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

# Enable CORS for frontend requests (e.g., GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client safely
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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


# Incoming Patient Request Schema
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


# OpenAI Structured Output Schema
class ClinicalInsight(BaseModel):
    summary_notes: str = Field(description="A 2-sentence clinical summary of the patient's condition based on vitals.")
    primary_recommendations: list[str] = Field(description="List of 2-3 specific actionable lifestyle or clinical changes.")
    follow_up_urgency: str = Field(description="Level of follow-up required: 'Routine', 'Urgent', or 'Immediate'")


@app.post("/predict")
def predict(payload: HeartData):
    # 1. Dynamic Risk Score Calculation based on payload vitals
    base_score = (payload.Age * 0.35) + (payload.RestingBP * 0.25) + (payload.Cholesterol * 0.15) + (payload.Oldpeak * 5.0)
    
    # Adjust score based on categorical risk factors
    if payload.FastingBS == 1:
        base_score += 10
    if payload.ExerciseAngina.lower() == "yes":
        base_score += 15
        
    # Scale score into a realistic percentage bounds (10.0% to 95.0%)
    probability = round(min(max(base_score / 2.8, 10.0), 95.0), 1)
    risk_level = "High" if probability >= 50 else "Low"

    # 2. Generate Structured Response via OpenAI (Optional enrichment step)
    ai_analysis = None
    if openai_client:
        try:
            completion = openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a clinical decision support assistant analyzing patient vitals and cardiac risk scores."
                    },
                    {
                        "role": "user", 
                        "content": f"Patient Age: {payload.Age}, Resting BP: {payload.RestingBP}, Cholesterol: {payload.Cholesterol}, Risk Score: {probability}% ({risk_level} Risk). Generate structured clinical notes."
                    }
                ],
                response_format=ClinicalInsight,
            )
            ai_analysis = completion.choices[0].message.parsed.model_dump()
        except Exception as e:
            print(f"OpenAI Generation Error: {e}")

    # 3. Save incoming assessment to PostgreSQL database
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

    # 4. Return combined key-values to UI
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