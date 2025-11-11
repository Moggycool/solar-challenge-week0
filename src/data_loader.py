"""
data_loader.py
---------------
Module for loading solar datasets from the local 'data/data' folder.
"""

from pathlib import Path
import pandas as pd

# Set your base data folder
BASE_DATA_DIR = Path(r"D:\Python\Week_01\data\data")


def load_data(country):
    """
    Loads a country's solar dataset (Togo, Sierra Leone, or Benin) from CSV.

    Args:
        country (str): One of ["togo", "sierraleone", "benin"]

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    country = country.lower()
    file_map = {
        "togo": "togo-dapaong_qc.csv",
        "sierraleone": "sierraleone-bumbuna.csv",
        "benin": "benin-malanville.csv"
    }

    if country not in file_map:
        raise ValueError(
            f"Invalid country name: {country}. Choose from {list(file_map.keys())}")

    file_path = BASE_DATA_DIR / file_map[country]

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"📂 Loading dataset for {country.title()} from {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    return df


def save_data(df, country, suffix="_clean"):
    """
    Saves a cleaned dataset to the same folder with a suffix.
    Example: benin_clean.csv
    """
    output_name = f"{country.lower()}{suffix}.csv"
    output_path = BASE_DATA_DIR / output_name
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")
