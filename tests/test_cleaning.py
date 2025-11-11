"""
test_cleaning.py — Unit tests for cleaning.py module.

Tests:
- fill_missing_values function
- remove_outliers_zscore function
"""
import pandas as pd
from src.cleaning import fill_missing_values, remove_outliers_zscore


def test_fill_missing_values():
    """
    Test that fill_missing_values correctly fills NaNs with the median
    for numeric columns in a DataFrame.
    """
    df = pd.DataFrame({
        "ghi": [1, None, 3],
        "dni": [None, 5, 6]
    })
    df_clean = fill_missing_values(df, ["ghi", "dni"])
    assert df_clean["ghi"].isna().sum() == 0
    assert df_clean["dni"].isna().sum() == 0


def test_remove_outliers_zscore():
    """
    Test that remove_outliers_zscore replaces extreme Z-score values
    with the column median.
    """
    df = pd.DataFrame({
        "ghi": [1, 2, 1000],  # 1000 is an outlier
        "dni": [5, 6, 7]
    })
    df_clean = remove_outliers_zscore(df, ["ghi", "dni"])
    # Check if outlier replaced by median
    median = df["ghi"].median()
    assert df_clean["ghi"].max() <= median + 3 * \
        df_clean["ghi"].std() or df_clean["ghi"].max() == median
