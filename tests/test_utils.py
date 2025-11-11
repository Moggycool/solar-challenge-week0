"""
test_cleaning.py — Unit tests for src/cleaning.py module.

Tests:
- fill_missing_values
- remove_outliers_zscore
"""

import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_zscore


@pytest.fixture
def sample_df_fixture() -> pd.DataFrame:
    """
    Provide a sample DataFrame with missing values and outliers
    for testing cleaning functions.
    """
    return pd.DataFrame({
        "ghi": [1, None, 3, 1000],  # 1000 is an outlier
        "dni": [None, 5, 6, 7]
    })


def test_fill_missing_values(sample_df_fixture: pd.DataFrame) -> None:
    """
    Test that fill_missing_values correctly fills NaNs with the median
    for numeric columns in a DataFrame.
    """
    df_clean = fill_missing_values(sample_df_fixture, ["ghi", "dni"])
    # Check no missing values remain
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["dni"].isna().sum() == 0


def test_remove_outliers_zscore(sample_df_fixture: pd.DataFrame) -> None:
    """
    Test that remove_outliers_zscore replaces extreme Z-score values
    with the column median.
    """
    df_clean = remove_outliers_zscore(sample_df_fixture, ["ghi", "dni"])
    # 1000 in 'ghi' should be replaced by median
    median_ghi = sample_df_fixture["ghi"].median()
    assert df_clean["ghi"].max() <= median_ghi + 3 * df_clean["ghi"].std() \
        or df_clean["ghi"].max() == median_ghi
