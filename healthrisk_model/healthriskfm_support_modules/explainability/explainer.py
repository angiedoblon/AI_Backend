import shap
import pandas as pd
import numpy as np
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)

class SHAPExplainer:
    """
    Computes SHAP values to explain model predictions.
    """
    def __init__(self, model: Any, X_background: pd.DataFrame):
        self.model = model
        self.X_background = X_background

        logger.info("Initializing SHAP explainer...")
        try:
            # Tree explainer is faster for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # Fallback to general explainer
            self.explainer = shap.Explainer(self.model, self.X_background)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculates SHAP values for a given dataset.
        """
        logger.info("Calculating SHAP values...")
        shap_values = self.explainer(X)
        return shap_values

    def explain_instance(self, X_instance: pd.DataFrame) -> dict:
        """
        Explains a single prediction.
        """
        shap_values = self.explainer(X_instance)
        # Assuming binary classification, get values for the positive class (often index 1 or the only output)
        vals = shap_values.values[0]
        if vals.ndim > 1:
            vals = vals[:, 1]

        feature_importance = dict(zip(X_instance.columns, vals))

        # Sort by absolute impact
        sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        return sorted_importance
