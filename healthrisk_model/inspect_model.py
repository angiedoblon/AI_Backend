
import sys
from pathlib import Path
import joblib

# Locate the model folder and support modules
BASE_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = BASE_DIR / "healthriskfm_support_modules"

# Make preprocessing, explainability, and utils importable
sys.path.insert(0, str(SUPPORT_DIR))

print("Loading HealthRiskFM model...")

pipeline = joblib.load(BASE_DIR / "pipeline.joblib")
model = joblib.load(BASE_DIR / "model.joblib")

print("\n===== PIPELINE INFORMATION =====")
print("Pipeline type:", type(pipeline))
print("Pipeline steps:", getattr(pipeline, "steps", "Not available"))
print("Expected input features:", getattr(pipeline, "feature_names_in_", "Not available"))
print("Expected input count:", getattr(pipeline, "n_features_in_", "Not available"))

print("\n===== MODEL INFORMATION =====")
print("Model type:", type(model))
print("Model classes:", getattr(model, "classes_", "Not available"))
print("Model input features:", getattr(model, "feature_names_in_", "Not available"))
print("Model input count:", getattr(model, "n_features_in_", "Not available"))
print("Has predict_proba:", hasattr(model, "predict_proba"))

print("\nInspection completed.")
