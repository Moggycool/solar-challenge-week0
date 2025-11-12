# Common helper functions
"""
utils.py — Shared utility functions for solar dataset analysis.

Provides reusable helpers for file operations, logging, data validation,
and visualization consistency.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# 1️⃣ File Handling Utilities
# ---------------------------------------------------------------------
def ensure_directory(path: str) -> None:
    """Ensure a directory exists; create it if missing."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_csv_safely(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV safely, ensuring directory exists."""
    directory = os.path.dirname(path)
    ensure_directory(directory)

    try:
        df.to_csv(path, index=False)
        print(f"✅ Saved file: {path}")
    except (OSError, ValueError) as err:
        print(f"⚠️ Failed to save CSV: {err}")


def load_csv_safely(path: str) -> pd.DataFrame:
    """Load a CSV safely, returning an empty DataFrame on failure."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded file: {path}")
        return df
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"⚠️ File is empty: {path}")
        return pd.DataFrame()
    except pd.errors.ParserError as err:
        print(f"⚠️ Parsing error while reading {path}: {err}")
        return pd.DataFrame()


# ---------------------------------------------------------------------
# 2️⃣ Logging and Info Utilities
# ---------------------------------------------------------------------
def print_section_header(title: str) -> None:
    """Print a styled section header for console clarity."""
    print("\n" + "=" * 80)
    print(f"{title.upper():^80}")
    print("=" * 80)


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print basic information about a DataFrame."""
    print_section_header(f"SUMMARY: {name}")
    print(f"Shape: {df.shape}")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())


# ---------------------------------------------------------------------
# 3️⃣ Data Validation Utilities
# ---------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """Check if all required columns exist in DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return False
    print("✅ All required columns found.")
    return True


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, logging how many were removed."""
    initial = len(df)
    df = df.drop_duplicates()
    removed = initial - len(df)
    if removed > 0:
        print(f"✅ Removed {removed} duplicate rows.")
    else:
        print("No duplicate rows found.")
    return df


# ---------------------------------------------------------------------
# 4️⃣ Plot Styling Utilities
# ---------------------------------------------------------------------
def set_plot_style() -> None:
    """Set a consistent Seaborn and Matplotlib plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.autolayout": True,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })


def save_plot(filename: str, folder: str = "plots") -> None:
    """Save current Matplotlib plot to a folder."""
    ensure_directory(folder)
    path = os.path.join(folder, filename)
    try:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"📊 Plot saved to: {path}")
    except (OSError, ValueError) as err:
        print(f"⚠️ Could not save plot: {err}")


# ---------------------------------------------------------------------
# 5️⃣ Utility for consistent naming
# ---------------------------------------------------------------------
def generate_clean_filename(prefix: str, country: str) -> str:
    if prefix:
        return f"data/{prefix}_{country}_clean.csv"
    else:
        return f"data/{country}_clean.csv"


# ---------------------------------------------------------------------
# Example usage (not executed when imported)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("utils.py provides helper functions — import it in other scripts, don’t run directly.")
