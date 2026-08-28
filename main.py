import os
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# Pydantic schema for incoming patient payload
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

@app.post("/predict")
def predict(payload: HeartData):
    # Perform prediction logic / calculation
    probability = 51.8  # Replace/bind with your model evaluation output
    risk_level = "High" if probability >= 50 else "Low"

    # Save incoming assessment to PostgreSQL database
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

    # Return key-values matched with client JS interface
    return {
        "heart_disease_risk": 1 if probability >= 50 else 0,
        "risk_probability_percent": probability,
        "risk_level": risk_level
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