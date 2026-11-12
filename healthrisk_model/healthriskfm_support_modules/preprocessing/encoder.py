import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)

class CategoricalEncoder:
    """
    Handles encoding of categorical variables.
    """
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.one_hot_encoder = None
        self.categorical_columns: List[str] = []
        self.ohe_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        self.categorical_columns = df_encoded.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if not self.categorical_columns:
            logger.info("No categorical columns to encode.")
            return df_encoded

        # For simplicity, if unique values <= 2, use LabelEncoder. Otherwise, use OneHotEncoder.
        binary_cols = [col for col in self.categorical_columns if df_encoded[col].nunique() <= 2]
        multi_cols = [col for col in self.categorical_columns if df_encoded[col].nunique() > 2]

        # Process binary
        for col in binary_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le

        # Process multi-class categoricals
        if multi_cols:
            self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_data = self.one_hot_encoder.fit_transform(df_encoded[multi_cols])
            self.ohe_columns = self.one_hot_encoder.get_feature_names_out(multi_cols)
            ohe_df = pd.DataFrame(ohe_data, columns=self.ohe_columns, index=df_encoded.index)

            df_encoded = pd.concat([df_encoded.drop(columns=multi_cols), ohe_df], axis=1)

        logger.info(f"Encoded {len(binary_cols)} binary columns and {len(multi_cols)} multi-class columns.")
        return df_encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()

        # Apply binary
        for col, le in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen labels by mapping to a known label or -1 (Requires custom handling, simplified here)
                df_encoded[col] = df_encoded[col].astype(str).map(
                    lambda s: s if s in le.classes_ else le.classes_[0] # simplified unseen handling
                )
                df_encoded[col] = le.transform(df_encoded[col])

        # Apply multi-class
        multi_cols = [col for col in self.categorical_columns if col not in self.label_encoders]
        if multi_cols and self.one_hot_encoder:
            # Only transform columns present in the original dataset
            cols_to_transform = [c for c in multi_cols if c in df_encoded.columns]
            ohe_data = self.one_hot_encoder.transform(df_encoded[cols_to_transform])
            ohe_df = pd.DataFrame(ohe_data, columns=self.ohe_columns, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=cols_to_transform), ohe_df], axis=1)

        return df_encoded
