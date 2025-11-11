"""
analyze.py — Exploratory analysis for solar datasets.

Includes:
- Summary statistics
- Missing value report
- Correlation analysis
- Time series plots
- Wind rose plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import set_plot_style, print_section_header

try:
    from windrose import WindroseAxes
    _HAS_WINDROSE = True
except ImportError:
    _HAS_WINDROSE = False


def summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics and missing values."""
    print_section_header("SUMMARY STATISTICS")
    print(df.describe(include="all", datetime_is_numeric=True))
    print("\nMissing values:")
    print(df.isna().sum())


def correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric columns."""
    set_plot_style()
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True)
    plt.title("Correlation Heatmap")
    plt.show()


def plot_time_series(df: pd.DataFrame, timestamp_col: str, cols: list[str]) -> None:
    """Plot time series for selected columns."""
    if timestamp_col not in df.columns:
        print(f"Timestamp column '{timestamp_col}' not found.")
        return

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)

    for col in cols:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            df[col].plot(title=f"{col.upper()} over Time")
            plt.xlabel("Time")
            plt.ylabel(col.upper())
            plt.tight_layout()
            plt.show()


def plot_wind_rose(df: pd.DataFrame, ws_col: str = "ws", wd_col: str = "wd") -> None:
    """Plot wind rose if Windrose library is installed."""
    if not _HAS_WINDROSE:
        print("Windrose library not installed. Skipping wind rose plot.")
        return

    try:
        ax = WindroseAxes.from_ax()
        ax.bar(df[wd_col], df[ws_col], normed=True,
               opening=0.8, edgecolor="white")
        ax.set_title("Wind Rose")
        plt.show()
    except KeyError as e:
        print(f"Columns missing for wind rose: {e}")


def run_full_analysis(df: pd.DataFrame, country: str, timestamp_col: str = "timestamp") -> None:
    """Run full EDA/analysis pipeline."""
    print_section_header(f"EDA for {country}")
    summary_statistics(df)
    correlation_heatmap(df)
    plot_time_series(df, timestamp_col, ["ghi", "dni", "dhi", "tamb"])
    if "ws" in df.columns and "wd" in df.columns:
        plot_wind_rose(df)
