"""
test_cleaning.py — Unit tests for src/cleaning.py module.

Tests:
- fill_missing_values
- remove_outliers_iqr
"""

import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_iqr

# pylint: disable=redefined-outer-name


@pytest.fixture
def sample_cleaning_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name
    """
    Provide a sample DataFrame with missing values and outliers
    for testing cleaning functions.
    """
    return pd.DataFrame({
        "ghi": [1, None, 3, 1000],  # 1000 is an outlier
        "dni": [None, 5, 6, 7]
    })


@pytest.fixture
def simple_outlier_df() -> pd.DataFrame:
    """Simple DataFrame with obvious outlier, no NaN values."""
    return pd.DataFrame({
        "ghi": [1, 2, 3, 1000],  # 1000 is clearly an outlier
        "dni": [4, 5, 6, 7]
    })


def test_fill_missing_values_numeric(sample_cleaning_df):
    """Test filling missing numeric values."""
    # Test with single column
    df_filled = fill_missing_values(sample_cleaning_df.copy(), ["ghi"])
    assert df_filled["ghi"].isna().sum() == 0
    # Should be filled with median (3.0) of [1, 3, 1000]
    assert df_filled.loc[1, "ghi"] == 3.0


def test_fill_missing_values_multiple_columns(sample_cleaning_df):
    """Test filling missing values in multiple columns."""
    df_filled = fill_missing_values(sample_cleaning_df.copy(), ["ghi", "dni"])
    assert df_filled["ghi"].isna().sum() == 0
    assert df_filled["dni"].isna().sum() == 0
    assert df_filled.loc[1, "ghi"] == 3.0  # median of [1, 3, 1000]
    assert df_filled.loc[0, "dni"] == 6.0  # median of [5, 6, 7]


def test_remove_outliers_iqr_debug(simple_outlier_df):
    """Test removing outliers with IQR method."""
    print("=== DEBUGGING remove_outliers_iqr ===")
    print("Original DataFrame:")
    print(simple_outlier_df)
    print("GHI values:", simple_outlier_df["ghi"].values)

    df_clean = remove_outliers_iqr(simple_outlier_df.copy(), ["ghi"])

    print("After remove_outliers_iqr:")
    print(df_clean)
    print("GHI values:", df_clean["ghi"].values)

    # The function REPLACES outliers with median, doesn't remove rows
    assert len(df_clean) == len(simple_outlier_df)

    # The outlier value 1000 should be REPLACED with the median (2.0)
    assert 1000 not in df_clean["ghi"].values
    assert df_clean.loc[3, "ghi"] == 2.0  # median of [1, 2, 3]


def test_remove_outliers_iqr_original(sample_cleaning_df):
    """Test removing outliers with original data (with NaN values)."""
    df_clean = remove_outliers_iqr(sample_cleaning_df.copy(), ["ghi"])

    # The function REPLACES outliers with median, doesn't remove rows
    assert len(df_clean) == len(sample_cleaning_df)

    # The outlier value 1000 should be REPLACED with the median
    assert 1000 not in df_clean["ghi"].values
