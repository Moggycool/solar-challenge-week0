"""
test_cleaning.py — Unit tests for cleaning.py module.
"""
import os
import sys
import pytest
import pandas as pd

from src.cleaning import fill_missing_values, remove_outliers_zscore, handle_missing

# Add the parent directory to Python path so src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# -----------------------------
# Fixture: Sample DataFrame
# -----------------------------
@pytest.fixture
def sample_cleaning_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name
    """Provides a sample DataFrame with numeric and categorical columns for cleaning tests."""
    return pd.DataFrame({
        "ghi": [1, None, 3, 1000],  # 1000 is an outlier
        "dni": [None, 5, 6, 7],
        "category": ["A", None, "B", "B"]
    })


# -----------------------------
# Tests for fill_missing_values
# -----------------------------
def test_fill_missing_values_numeric(sample_cleaning_df_fixture):
    """Test filling missing numeric values."""
    # Test with single column
    df_filled = fill_missing_values(sample_cleaning_df_fixture.copy(), ["ghi"])
    assert df_filled["ghi"].isna().sum() == 0
    # Should be filled with median (2.0) of [1, 3]
    assert df_filled.loc[1, "ghi"] == 2.0


def test_fill_missing_values_multiple_columns(sample_cleaning_df_fixture):
    """Test filling missing values in multiple columns."""
    df_filled = fill_missing_values(
        sample_cleaning_df_fixture.copy(), ["ghi", "dni"])
    assert df_filled["ghi"].isna().sum() == 0
    assert df_filled["dni"].isna().sum() == 0
    assert df_filled.loc[1, "ghi"] == 2.0  # median of [1, 3]
    assert df_filled.loc[0, "dni"] == 6.0  # median of [5, 6, 7]


def test_fill_missing_values_categorical_untouched(sample_cleaning_df_fixture):
    """Test that categorical columns are not modified."""
    df_filled = fill_missing_values(
        sample_cleaning_df_fixture.copy(), ["ghi", "dni"])
    # Category column should still have missing values since we didn't process it
    assert df_filled["category"].isna().sum() == 1


# -----------------------------
# Tests for remove_outliers_zscore
# -----------------------------
def test_remove_outliers_zscore(sample_cleaning_df_fixture):
    """Test removing outliers using Z-score method."""
    df_clean = remove_outliers_zscore(sample_cleaning_df_fixture.copy(), "ghi")
    # The row with ghi=1000 should be removed
    assert len(df_clean) == 3
    assert 1000 not in df_clean["ghi"].values


# -----------------------------
# Tests for handle_missing
# -----------------------------
def test_handle_missing_basic(sample_cleaning_df_fixture):
    """Test handle_missing fills all missing values."""
    df_clean = handle_missing(sample_cleaning_df_fixture.copy())

    # Should have no missing values
    assert df_clean.isna().sum().sum() == 0

    # Check numeric columns filled with median
    assert df_clean.loc[1, "ghi"] == 2.0  # median of [1, 3]
    assert df_clean.loc[0, "dni"] == 6.0  # median of [5, 6, 7]

    # Check categorical column filled with mode
    assert df_clean.loc[1, "category"] == "B"  # mode is "B"


def test_handle_missing_preserves_data_shape(sample_cleaning_df_fixture):
    """Test that handle_missing preserves DataFrame shape."""
    original_shape = sample_cleaning_df_fixture.shape
    df_clean = handle_missing(sample_cleaning_df_fixture.copy())

    # Should have same number of rows and columns
    assert df_clean.shape == original_shape
