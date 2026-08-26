import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Database
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age REAL,
            bmi REAL,
            blood_pressure REAL,
            risk_score REAL,
            risk_level TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()


# Input Schema
class PatientData(BaseModel):
    age: float
    bmi: float
    blood_pressure: float


@app.get("/")
def home():
    return {"status": "AI_Backend is live and running!"}


@app.post("/predict")
def predict(data: PatientData):
    # Dummy calculation logic (replace with your joblib model scoring)
    risk_score = round((data.age * 0.2) + (data.bmi * 0.5) + (data.blood_pressure * 0.3), 2)
    risk_level = "High" if risk_score > 70 else "Moderate" if risk_score > 40 else "Low"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to SQLite Database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (timestamp, age, bmi, blood_pressure, risk_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (timestamp, data.age, data.bmi, data.blood_pressure, risk_score, risk_level),
    )
    conn.commit()
    conn.close()

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "timestamp": timestamp,
    }


@app.get("/history")
def get_history():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, age, bmi, blood_pressure, risk_score, risk_level FROM predictions ORDER BY id DESC LIMIT 10"
    )
    rows = cursor.fetchall()
    conn.close()

    history = [
        {
            "timestamp": row[0],
            "age": row[1],
            "bmi": row[2],
            "blood_pressure": row[3],
            "risk_score": row[4],
            "risk_level": row[5],
        }
        for row in rows
    ]

    return {"history": history}