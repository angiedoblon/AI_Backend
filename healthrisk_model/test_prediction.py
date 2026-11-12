
import joblib
import pandas as pd

print("Loading HealthRiskFM...")

pipeline = joblib.load("pipeline.joblib")
model = joblib.load("model.joblib")

# Sample patient information
patient = pd.DataFrame([{
    "Age": 45,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 120,
    "Cholesterol": 220,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 1.0,
    "ST_Slope": "Up"
}])

print("\nOriginal patient data:")
print(patient.to_string(index=False))

# Apply the original trained preprocessing pipeline
processed = pipeline.transform(patient)

# Match the exact feature order expected by the model
expected_features = list(model.feature_names_in_)

missing = set(expected_features) - set(processed.columns)

if missing:
    raise ValueError(f"Missing model features: {missing}")

processed = processed.reindex(columns=expected_features)

print("\nProcessed feature count:", processed.shape[1])
print("Processed feature names:", list(processed.columns))

# Generate actual Random Forest prediction
prediction = model.predict(processed)[0]
probabilities = model.predict_proba(processed)[0]

print("\n===== HEALTHRISKFM PREDICTION =====")
print("Predicted class:", prediction)
print("Class labels:", model.classes_)

for label, probability in zip(model.classes_, probabilities):
    print(f"Class {label} probability: {probability * 100:.2f}%")
