"""
preprocess.py — Data preprocessing for solar datasets.

Handles:
- Column standardization
- Missing value imputation
- Outlier detection using Z-score
- Exporting cleaned datasets
"""

import pandas as pd
# import numpy as np
from src.utils import save_csv_safely, generate_clean_filename


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase and replace spaces with underscores."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def fill_missing_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values in numeric columns with median."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median())
    return df


def remove_outliers_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace outliers (|Z| > 3) with median in numeric columns."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            df.loc[z_scores.abs() > 3, col] = df[col].median()
    return df


def preprocess_dataset(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Run full preprocessing pipeline."""
    df = standardize_columns(df)

    key_cols = ["ghi", "dni", "dhi", "moda",
                "modb", "ws", "wsgust", "tamb", "rh"]
    df = fill_missing_values(df, key_cols)
    df = remove_outliers_zscore(df, key_cols)

    # Save cleaned dataset
    out_file = generate_clean_filename("dataset", country)
    save_csv_safely(df, out_file)

    return df
