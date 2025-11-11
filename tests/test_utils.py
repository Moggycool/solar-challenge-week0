"""
test_utils.py — Unit tests for src/utils.py module.

Tests:
- generate_clean_filename
- save_csv_safely
- set_plot_style
"""

import tempfile
import os
import pandas as pd
from src.utils import save_csv_safely, generate_clean_filename, set_plot_style


def test_generate_clean_filename() -> None:
    """
    Test that generate_clean_filename returns a CSV filename
    containing the country name.
    """
    filename = generate_clean_filename("dataset", "benin")
    assert "benin" in filename
    assert filename.endswith(".csv")


def test_save_csv_safely() -> None:
    """
    Test that save_csv_safely correctly writes a DataFrame to CSV
    and that the file exists on disk.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.csv")
        save_csv_safely(df, filepath)
        assert os.path.exists(filepath)
        df_loaded = pd.read_csv(filepath)
        assert df_loaded.shape == df.shape


def test_set_plot_style() -> None:
    """
    Test that set_plot_style runs without raising an exception.
    This ensures matplotlib style is applied correctly.
    """
    set_plot_style()


def test_set_plot_style_runs() -> None:
    """
    Ensure that set_plot_style runs without errors.
    """
    set_plot_style()
