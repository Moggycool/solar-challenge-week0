"""
test_data_loader.py — Unit tests for data_loader.py module.

Uses pytest to verify:
- CSV loading
- DataFrame shape and column correctness
"""
# pylint: disable=unused-import
import pytest
import pandas as pd
from src.data_loader import load_data


def test_load_country_data(tmp_path):
    """
    Test loading CSV files into a DataFrame.
    """
    # Create a dummy CSV
    dummy_csv = tmp_path / "dummy.csv"
    df_input = pd.DataFrame({
        "ghi": [1, 2, 3],
        "dni": [4, 5, 6],
        "dhi": [7, 8, 9]
    })
    df_input.to_csv(dummy_csv, index=False)

    # Load using function
    df_loaded = load_data(str(dummy_csv))
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (3, 3)
    assert list(df_loaded.columns) == ["ghi", "dni", "dhi"]
