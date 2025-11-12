"""
test_cleaning.py — Unit tests for src/cleaning.py module.

Tests:
- fill_missing_values
- remove_outliers_zscore
"""

import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_zscore

# pylint: disable=redefined-outer-name


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


def test_remove_outliers_zscore_simple(simple_outlier_df):
    """Test removing outliers with simple data (no NaN values)."""
    print("Simple test case - no NaN values")
    print("Original:", simple_outlier_df["ghi"].values)

    df_clean = remove_outliers_zscore(simple_outlier_df.copy(), ["ghi"])

    print("After cleaning:", df_clean["ghi"].values)

    # Calculate what should happen
    ghi_values = simple_outlier_df["ghi"].values  # [1, 2, 3, 1000]
    mean = ghi_values.mean()  # (1+2+3+1000)/4 = 1006/4 = 251.5
    std = ghi_values.std()    # Large due to outlier
    z_scores = (ghi_values - mean) / std
    print(f"Mean: {mean}, Std: {std}")
    print("Z-scores:", z_scores)

    # The outlier should be replaced with median (2.0)
    assert 1000 not in df_clean["ghi"].values
    assert df_clean.loc[3, "ghi"] == 2.0  # median of [1, 2, 3]
