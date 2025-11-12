"""
test_utils.py — Unit tests for utility functions.
"""

import pandas as pd
import pytest
from src.cleaning import fill_missing_values, remove_outliers_iqr

# pylint: disable=redefined-outer-name


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame({
        "column1": [1, None, 3, 1000],
        "column2": [None, 5, 6, 7]
    })


def test_fill_missing_values_basic(sample_df):
    """Test basic missing value filling."""
    df_filled = fill_missing_values(sample_df.copy(), ["column1"])
    assert df_filled["column1"].isna().sum() == 0


def test_remove_outliers_iqr_basic(sample_df):
    """Test basic outlier removal with IQR method."""
    df_clean = remove_outliers_iqr(sample_df.copy(), ["column1"])
    assert len(df_clean) == len(sample_df)
    # Outlier should be replaced
    assert 1000 not in df_clean["column1"].values
