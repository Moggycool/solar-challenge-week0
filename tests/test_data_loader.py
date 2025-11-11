"""
test_data_loader.py — Unit tests for src/data_loader.py module.

Tests:
- load_country_data function
"""

import pandas as pd
from src import data_loader
from src.data_loader import load_country_data


def test_load_country_data(tmp_path: str) -> None:
    """
    Test that load_country_data correctly loads a CSV from the specified directory,
    strips column names, and returns a DataFrame.
    """
    # Create a dummy CSV in a temporary directory
    dummy_csv = tmp_path / "dummy.csv"
    df_input = pd.DataFrame({
        "ghi": [1, 2, 3],
        "dni": [4, 5, 6],
        "dhi": [7, 8, 9]
    })
    df_input.to_csv(dummy_csv, index=False)

    # Temporarily override BASE_DATA_DIR for testing
    original_base = data_loader.BASE_DATA_DIR
    data_loader.BASE_DATA_DIR = tmp_path

    try:
        df_loaded = load_country_data("dummy.csv")
        assert isinstance(df_loaded, pd.DataFrame)
        assert df_loaded.shape == (3, 3)
        assert list(df_loaded.columns) == ["ghi", "dni", "dhi"]
    finally:
        # Restore original BASE_DATA_DIR
        data_loader.BASE_DATA_DIR = original_base
