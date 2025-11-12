"""
Unit tests for src/data_loader.py module.

Covers:
- Successful loading of existing data
- Handling of missing or invalid files
- Behavior with empty files
"""

import pandas as pd
import pytest
from src import data_loader
from src.data_loader import load_country_data


def test_load_country_data_valid(monkeypatch, tmp_path):
    """Test loading a valid CSV file returns a non-empty DataFrame."""
    # Create a dummy CSV file
    sample_data = pd.DataFrame({
        "Country": ["Benin", "Togo"],
        "Value": [100, 200]
    })
    csv_path = tmp_path / "benin.csv"
    sample_data.to_csv(csv_path, index=False)

    # Monkeypatch the loader to use this file
    monkeypatch.setattr(data_loader, "DATA_PATH", tmp_path)

    df = load_country_data("benin")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Country" in df.columns


def test_load_country_data_missing_file(monkeypatch, tmp_path):
    """Test loading a non-existent country raises FileNotFoundError."""
    monkeypatch.setattr(data_loader, "DATA_PATH", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_country_data("missing_country")


def test_load_country_data_empty_file(monkeypatch, tmp_path):
    """Test loading an empty CSV file returns an empty DataFrame."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")  # create an empty file

    # Patch so it looks for this file
    monkeypatch.setattr(data_loader, "DATA_PATH", tmp_path)

    with pytest.raises(pd.errors.EmptyDataError):
        load_country_data("empty")
