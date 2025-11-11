"""
test_cleaning.py — Unit tests for cleaning.py module.

Tests:
- fill_missing_values function
- remove_outliers_zscore function
- handle_missing function
"""
import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_zscore, handle_missing


# -----------------------------
# Fixture: Sample DataFrame
# -----------------------------
@pytest.fixture
def sample_df_fixture() -> pd.DataFrame:
    """Provides a sample DataFrame with numeric and categorical columns for testing."""
    return pd.DataFrame({
        "ghi": [1, None, 3, 1000],    # 1000 is an outlier
        "dni": [None, 5, 6, 7],
        "category": ["A", None, "B", "B"]
    })


# -----------------------------
# Test fill_missing_values
# -----------------------------
def test_fill_missing_values(sample_df_fixture: pd.DataFrame) -> None:
    """Test that fill_missing_values correctly fills NaNs with median values for numeric columns."""
    df_clean = fill_missing_values(sample_df_fixture, ["ghi", "dni"])

    # Numeric columns should have no NaNs
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["dni"].isna().sum() == 0

    # Median filling check
    expected_ghi_median = pd.Series([1, 2, 3, 1000]).median()
    expected_dni_median = pd.Series([5, 5, 6, 7]).median()
    assert df_clean.loc[1, "ghi"] == expected_ghi_median
    assert df_clean.loc[0, "dni"] == expected_dni_median


# -----------------------------
# Test remove_outliers_zscore
# -----------------------------
def test_remove_outliers_zscore(sample_df_fixture: pd.DataFrame) -> None:
    """Test that remove_outliers_zscore replaces extreme Z-score values with the column median."""
    df_clean = remove_outliers_zscore(sample_df_fixture, ["ghi"])

    # Outlier 1000 should be replaced by median
    median_val = df_clean["ghi"].median()
    assert df_clean["ghi"].max() <= 1000
    assert 1000 not in df_clean["ghi"].values
    # All other values remain unchanged
    assert set(df_clean["ghi"].dropna()) <= {1, 3, median_val}


# -----------------------------
# Test handle_missing
# -----------------------------
def test_handle_missing(sample_df_fixture: pd.DataFrame) -> None:
    """Test that handle_missing fills numeric NaNs with median and categorical NaNs with mode."""
    df_clean = handle_missing(sample_df_fixture)

    # Numeric columns filled
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["dni"].isna().sum() == 0

    # Categorical column filled with mode
    assert df_clean["category"].isna().sum() == 0
    assert df_clean.loc[1, "category"] == "B"  # mode of ["A","B","B"] = "B"
