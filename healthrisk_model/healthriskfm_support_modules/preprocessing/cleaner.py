import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.impute import SimpleImputer, KNNImputer

from utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """
    Handles missing values, outliers, and basic data cleaning tasks.
    """
    def __init__(self, imputation_strategy: str = 'median'):
        self.imputation_strategy = imputation_strategy
        self.imputers = {}

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        df_clean = df.copy()

        # 1. Drop duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_len:
            logger.info(f"Dropped {initial_len - len(df_clean)} duplicate rows.")

        # 2. Impute missing values
        features = df_clean.drop(columns=[target_col]) if target_col in df_clean else df_clean
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) > 0:
            if self.imputation_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=self.imputation_strategy)

            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
            self.imputers['numeric'] = imputer

        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = imputer.fit_transform(df_clean[categorical_cols])
            self.imputers['categorical'] = imputer

        return df_clean

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns

        if 'numeric' in self.imputers and len(numeric_cols) > 0:
            # Only impute columns that the imputer was trained on (or subset)
            cols = [c for c in numeric_cols if c in self.imputers['numeric'].feature_names_in_]
            if cols:
                # To be robust in production, handle exact columns
                df_clean[numeric_cols] = self.imputers['numeric'].transform(df_clean[numeric_cols])

        if 'categorical' in self.imputers and len(categorical_cols) > 0:
            df_clean[categorical_cols] = self.imputers['categorical'].transform(df_clean[categorical_cols])

        return df_clean
