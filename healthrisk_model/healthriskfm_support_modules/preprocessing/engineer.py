import pandas as pd
import numpy as np
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Generates engineered features and handles feature selection/filtering (e.g. constant columns).
    """
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.columns_to_drop = []

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df_engineered = df.copy()

        # 1. Identify constant columns
        constant_cols = [col for col in df_engineered.columns if df_engineered[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Dropping constant columns: {constant_cols}")
            self.columns_to_drop.extend(constant_cols)

        # 2. Identify highly correlated numerical features (to avoid multicollinearity)
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != target_col and c not in self.columns_to_drop]

        if len(numeric_cols) > 1:
            corr_matrix = df_engineered[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop_corr = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
            if to_drop_corr:
                logger.info(f"Dropping highly correlated columns (> {self.correlation_threshold}): {to_drop_corr}")
                self.columns_to_drop.extend(to_drop_corr)

        # Drop identified columns
        df_engineered = df_engineered.drop(columns=self.columns_to_drop, errors='ignore')

        # Generate some basic interaction features for remaining numeric cols if it's small enough
        # (Optional, skipped here for simplicity to keep generic, but can be added based on disease type)

        return df_engineered

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_engineered = df.copy()
        df_engineered = df_engineered.drop(columns=self.columns_to_drop, errors='ignore')
        return df_engineered
