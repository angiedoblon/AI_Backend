import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

class FeatureScaler:
    """
    Scales numerical features.
    """
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None
        self.numeric_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df_scaled = df.copy()
        exclude_cols = exclude_cols or []

        self.numeric_columns = [
            col for col in df_scaled.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        if not self.numeric_columns:
            logger.info("No numeric columns to scale.")
            return df_scaled

        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        df_scaled[self.numeric_columns] = self.scaler.fit_transform(df_scaled[self.numeric_columns])
        logger.info(f"Scaled {len(self.numeric_columns)} numeric columns using {self.method} scaler.")

        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()

        if self.scaler and self.numeric_columns:
            cols_to_scale = [c for c in self.numeric_columns if c in df_scaled.columns]
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])

        return df_scaled
