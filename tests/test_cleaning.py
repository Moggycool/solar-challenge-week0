"""
test_cleaning.py — Unit tests for cleaning.py module.
"""
import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_zscore, handle_missing


# -----------------------------
# Fixture: Sample DataFrame
# -----------------------------
@pytest.fixture
def df_sample_cleaning() -> pd.DataFrame:
    """Provides a sample DataFrame with numeric and categorical columns for cleaning tests."""
    return pd.DataFrame({
        "ghi": [1, None, 3, 1000],  # 1000 is an outlier
        "dni": [None, 5, 6, 7],
        "category": ["A", None, "B", "B"]
    })


# -----------------------------
# Tests
# -----------------------------
def test_fill_missing_values(df_sample_cleaning: pd.DataFrame) -> None:
    """Test fill_missing_values fills NaNs with median values."""
    df_clean = fill_missing_values(df_sample_cleaning, ["ghi", "dni"])
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["dni"].isna().sum() == 0


def test_remove_outliers_zscore(df_sample_cleaning: pd.DataFrame) -> None:
    """Test remove_outliers_zscore replaces extreme Z-score values."""
    df_clean = remove_outliers_zscore(df_sample_cleaning, ["ghi"])
    assert 1000 not in df_clean["ghi"].values


def test_handle_missing(df_sample_cleaning: pd.DataFrame) -> None:
    """Test handle_missing fills numeric and categorical NaNs."""
    df_clean = handle_missing(df_sample_cleaning)
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["category"].isna().sum() == 0
