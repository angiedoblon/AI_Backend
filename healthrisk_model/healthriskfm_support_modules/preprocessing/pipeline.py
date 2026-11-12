import pandas as pd
from typing import Tuple, Dict, Any

from preprocessing.cleaner import DataCleaner
from preprocessing.engineer import FeatureEngineer
from preprocessing.encoder import CategoricalEncoder
from preprocessing.scaler import FeatureScaler
from utils.logger import get_logger

logger = get_logger(__name__)

class PreprocessingPipeline:
    """
    Orchestrates the entire preprocessing workflow: cleaning, engineering, encoding, scaling.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_col = config.get('data', {}).get('target_column', 'target')

        imputation_strategy = config.get('preprocessing', {}).get('imputation_strategy', 'median')
        scaling_method = config.get('preprocessing', {}).get('scaling_method', 'standard')

        self.cleaner = DataCleaner(imputation_strategy=imputation_strategy)
        self.engineer = FeatureEngineer()
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler(method=scaling_method)

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting preprocessing pipeline fit_transform...")

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in the dataset.")

        y = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col])

        # 1. Clean (impute missing, handle duplicates)
        logger.info("Step 1/4: Cleaning data...")
        X_clean = self.cleaner.fit_transform(X)

        # 2. Engineer features
        logger.info("Step 2/4: Engineering features...")
        X_eng = self.engineer.fit_transform(X_clean, target_col=self.target_col)

        # 3. Encode categoricals
        logger.info("Step 3/4: Encoding categorical features...")
        X_enc = self.encoder.fit_transform(X_eng)

        # 4. Scale numericals
        logger.info("Step 4/4: Scaling numerical features...")
        X_scaled = self.scaler.fit_transform(X_enc)

        logger.info("Preprocessing complete.")
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing pipeline transform...")

        # Drop target column if present
        X = df.copy()
        if self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])

        # 1. Clean
        X = self.cleaner.transform(X)

        # 2. Engineer
        X = self.engineer.transform(X)

        # 3. Encode
        X = self.encoder.transform(X)

        # 4. Scale
        X = self.scaler.transform(X)

        logger.info("Transform complete.")
        return X
