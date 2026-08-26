import csv
from datetime import datetime
import io
import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Database with full schema
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER,
            sex TEXT,
            chest_pain_type TEXT,
            resting_bp INTEGER,
            cholesterol INTEGER,
            fasting_bs INTEGER,
            resting_ecg TEXT,
            max_hr INTEGER,
            exercise_angina TEXT,
            oldpeak REAL,
            st_slope TEXT,
            risk_score REAL,
            risk_level TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


# Input Schema - Matching exact keys sent by index.html
class PatientData(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Sex: str
    ChestPainType: str
    RestingBP: int = Field(..., ge=50, le=250)
    Cholesterol: int = Field(..., ge=50, le=600)
    FastingBS: int = Field(..., ge=0, le=1)
    RestingECG: str
    MaxHR: int = Field(..., ge=60, le=220)
    ExerciseAngina: str
    Oldpeak: float = Field(..., ge=0.0, le=10.0)
    ST_Slope: str


@app.get("/")
def home():
    return {"status": "AI_Backend is live and running!"}


@app.post("/predict")
def predict(data: PatientData):
    # Dummy calculation logic (replace with model.predict if using joblib)
    risk_score = round(
        (data.Age * 0.3)
        + (data.RestingBP * 0.2)
        + (data.Cholesterol * 0.1)
        + (data.Oldpeak * 5.0),
        1,
    )
    risk_probability = min(risk_score, 100.0)
    heart_disease_risk = 1 if risk_probability > 50.0 else 0
    risk_level = "High" if heart_disease_risk == 1 else "Low"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save complete record to SQLite Database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp, age, sex, chest_pain_type, resting_bp, cholesterol, 
            fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope, 
            risk_score, risk_level
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            timestamp,
            data.Age,
            data.Sex,
            data.ChestPainType,
            data.RestingBP,
            data.Cholesterol,
            data.FastingBS,
            data.RestingECG,
            data.MaxHR,
            data.ExerciseAngina,
            data.Oldpeak,
            data.ST_Slope,
            risk_probability,
            risk_level,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "risk_probability_percent": risk_probability,
        "heart_disease_risk": heart_disease_risk,
        "risk_level": risk_level,
        "timestamp": timestamp,
    }


@app.get("/history")
def get_history():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, age, resting_bp, cholesterol, risk_score, risk_level 
        FROM predictions 
        ORDER BY id DESC LIMIT 10
    """
    )
    rows = cursor.fetchall()
    conn.close()

    history = [
        {
            "timestamp": row[0],
            "age": row[1],
            "blood_pressure": row[2],
            "bmi": row[3],  # Mapping cholesterol field to display column
            "risk_score": row[4],
            "risk_level": row[5],
        }
        for row in rows
    ]

    return {"history": history}


@app.get("/export-csv")
def export_csv():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, age, sex, chest_pain_type, resting_bp, cholesterol, 
               fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope, 
               risk_score, risk_level 
        FROM predictions 
        ORDER BY id DESC
    """
    )
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    # Header Row
    writer.writerow(
        [
            "ID",
            "Timestamp",
            "Age",
            "Sex",
            "ChestPainType",
            "RestingBP",
            "Cholesterol",
            "FastingBS",
            "RestingECG",
            "MaxHR",
            "ExerciseAngina",
            "Oldpeak",
            "ST_Slope",
            "RiskScore",
            "RiskLevel",
        ]
    )

    # Data Rows
    for row in rows:
        writer.writerow(row)

    output.seek(0)

    headers = {"Content-Disposition": 'attachment; filename="patient_records.csv"'}
    return StreamingResponse(output, media_type="text/csv", headers=headers)