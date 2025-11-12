"""
cleaning.py
------------
Module for cleaning and preprocessing solar datasets.
"""

import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names: lowercase, underscores, no spaces.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows, if any.
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing numeric values with median and categorical with mode.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'O':  # object / categorical
            mode_val = df[col].mode(
            )[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_outliers(df: pd.DataFrame, cols: list[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers using the IQR (Interquartile Range) method.

    Args:
        df: pandas DataFrame
        cols: list of column names to check for outliers
        factor: IQR multiplier (default=1.5)
    """
    df = df.copy()
    for col in cols:
        if df[col].dtype != 'O':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def fill_missing_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Fill missing values in numeric columns with median (future-proof, no inplace warning).
    """
    df_filled = df.copy()
    for col in cols:
        if col in df_filled.columns:
            df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    return df_filled


def remove_outliers_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Replace outliers (|Z| > 3) with median (future-proof, avoids inplace warnings).
    """
    df_clean = df.copy()
    for col in cols:
        if col in df_clean.columns:
            # Convert to numeric and drop NaN for Z-score calculation
            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')

            # Calculate mean and std only on non-NaN values
            non_null_values = numeric_series.dropna()
            if len(non_null_values) > 0:  # Only process if we have valid values
                mean = non_null_values.mean()
                std = non_null_values.std()

                # Avoid division by zero
                if std > 0:
                    z_scores = (numeric_series - mean) / std
                    # Replace outliers with median (using non-NaN values for median)
                    median_val = non_null_values.median()
                    df_clean.loc[z_scores.abs() > 3, col] = median_val

    return df_clean
