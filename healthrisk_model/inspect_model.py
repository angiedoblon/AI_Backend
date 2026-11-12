
import joblib

print("Loading HealthRiskFM model...")

pipeline = joblib.load("pipeline.joblib")
model = joblib.load("model.joblib")

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
