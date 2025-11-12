"""
data_loader.py — Functions to load solar datasets from the workspace data folder.
"""

from pathlib import Path
import pandas as pd

# Base directory where raw CSV files are stored
BASE_DATA_DIR = Path(
    r"D:\Python\Week_01\Assignment\solar-challenge-week0\data")


def load_country_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the base data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file to load (e.g., "benin-malanville.csv").

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    file_path = BASE_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    return df
