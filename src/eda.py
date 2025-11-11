"""
eda.py — Exploratory Data Analysis module for solar dataset.

Performs statistical summaries, missing-value analysis, outlier detection,
and visualizations including correlation, distribution, and wind rose analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from windrose import WindroseAxes  # Optional dependency
    _HAS_WINDROSE = True
except ImportError:
    _HAS_WINDROSE = False


# ---------------------------------------------------------------------
# 1️⃣ Summary statistics and missing-value report
# ---------------------------------------------------------------------
def summary_and_missing_report(df: pd.DataFrame) -> None:
    """Display summary statistics and missing-value counts."""
    print("=== SUMMARY STATISTICS ===")
    print(df.describe(include="all", datetime_is_numeric=True))

    print("\n=== MISSING VALUE REPORT ===")
    missing_report = df.isna().sum()
    print(missing_report)

    null_threshold = len(df) * 0.05
    cols_over_5pct = missing_report[missing_report > null_threshold]

    if not cols_over_5pct.empty:
        print("\nColumns with >5% missing values:")
        print(cols_over_5pct)
    else:
        print("\nNo columns with >5% missing values.")


# ---------------------------------------------------------------------
# 2️⃣ Outlier detection and basic cleaning
# ---------------------------------------------------------------------
def detect_and_clean_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Detect and clean outliers using Z-score; impute missing values."""
    df_clean = df.copy()

    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            z_scores = (df_clean[col] - df_clean[col].mean()
                        ) / df_clean[col].std()
            df_clean.loc[z_scores.abs() > 3, col] = np.nan
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


# ---------------------------------------------------------------------
# 3️⃣ Time Series Analysis
# ---------------------------------------------------------------------
def time_series_analysis(df: pd.DataFrame, timestamp_col: str, cols: list[str]) -> None:
    """Plot time series for given columns against timestamp."""
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(by=timestamp_col)
        df.set_index(timestamp_col, inplace=True)
    except (KeyError, ValueError) as err:
        print(f"Time column issue: {err}")
        return

    for col in cols:
        if col in df.columns:
            plt.figure(figsize=(10, 4))
            df[col].plot(title=f"{col} over Time")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()


# ---------------------------------------------------------------------
# 4️⃣ Cleaning Impact
# ---------------------------------------------------------------------
def plot_cleaning_impact(
    df: pd.DataFrame,
    cleaning_flag_col: str = "cleaning",
    mod_cols: list[str] | None = None,
) -> None:
    """Plot average ModA and ModB values before/after cleaning."""
    if mod_cols is None:
        mod_cols = ["moda", "modb"]

    if cleaning_flag_col not in df.columns:
        print(f"Column {cleaning_flag_col} not found.")
        return

    try:
        grouped = df.groupby(cleaning_flag_col)[mod_cols].mean()
        grouped.plot(kind="bar", figsize=(6, 4))
        plt.title("Average ModA & ModB by Cleaning Flag")
        plt.ylabel("Average Value")
        plt.tight_layout()
        plt.show()
    except (KeyError, ValueError, TypeError) as err:
        print(f"Could not plot cleaning impact: {err}")


# ---------------------------------------------------------------------
# 5️⃣ Correlation & Relationship Analysis
# ---------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot heatmap of key correlations."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    except (KeyError, ValueError) as err:
        print(f"Correlation plot failed: {err}")


def scatter_plots(df: pd.DataFrame) -> None:
    """Generate selected scatter plots."""
    pairs = [
        ("ws", "ghi"),
        ("wsgust", "ghi"),
        ("wd", "ghi"),
        ("rh", "tamb"),
        ("rh", "ghi"),
    ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(5, 4))
            sns.scatterplot(data=df, x=x, y=y)
            plt.title(f"{x.upper()} vs {y.upper()}")
            plt.tight_layout()
            plt.show()


# ---------------------------------------------------------------------
# 6️⃣ Wind & Distribution Analysis
# ---------------------------------------------------------------------
def wind_rose_plot(df: pd.DataFrame, ws_col: str = "ws", wd_col: str = "wd") -> None:
    """Plot wind rose if windrose library is available."""
    if not _HAS_WINDROSE:
        print("Windrose library not installed; skipping wind plot.")
        return

    try:
        ax = WindroseAxes.from_ax()
        ax.bar(df[wd_col], df[ws_col], normed=True,
               opening=0.8, edgecolor="white")
        ax.set_title("Wind Rose Plot")
        plt.show()
    except (KeyError, ValueError) as err:
        print(f"Wind rose plot failed: {err}")


def histogram_and_distribution(df: pd.DataFrame, var: str) -> None:
    """Plot histogram for a given variable."""
    if var in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[var], bins=30, kde=True)
        plt.title(f"Distribution of {var}")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# 7️⃣ Temperature Analysis
# ---------------------------------------------------------------------
def temperature_vs_humidity_analysis(df: pd.DataFrame) -> None:
    """Analyze how RH influences temperature."""
    if "rh" in df.columns and "tamb" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df["rh"], y=df["tamb"])
        plt.title("Relative Humidity vs Temperature (Tamb)")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# 8️⃣ Bubble Chart
# ---------------------------------------------------------------------
def bubble_chart_ghi_vs_tamb(df: pd.DataFrame) -> None:
    """Plot bubble chart of GHI vs Tamb with bubble size = RH or BP."""
    if "ghi" in df.columns and "tamb" in df.columns:
        bubble_size = (
            df["rh"]
            if "rh" in df.columns
            else df.get("bp", pd.Series(20, index=df.index))
        )
        plt.figure(figsize=(6, 4))
        plt.scatter(df["ghi"], df["tamb"], s=bubble_size, alpha=0.5)
        plt.title("GHI vs Tamb (Bubble = RH or BP)")
        plt.xlabel("GHI")
        plt.ylabel("Tamb")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# 🧩 Full Pipeline Runner
# ---------------------------------------------------------------------
def run_full_eda_pipeline(
    df: pd.DataFrame,
    country: str,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Run the entire EDA pipeline."""
    print(f"=== EDA for {country.upper()} ===")

    summary_and_missing_report(df)

    numeric_cols = ["ghi", "dni", "dhi", "moda", "modb", "ws", "wsgust"]
    df_cleaned = detect_and_clean_outliers(df, numeric_cols)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", f"{country.lower()}_clean.csv")
    df_cleaned.to_csv(out_path, index=False)
    print(f"Cleaned dataset exported to: {out_path}")

    time_series_analysis(df_cleaned, timestamp_col, [
                         "ghi", "dni", "dhi", "tamb"])
    plot_cleaning_impact(df_cleaned, "cleaning", ["moda", "modb"])
    correlation_heatmap(df_cleaned)
    scatter_plots(df_cleaned)

    if "ws" in df_cleaned.columns and "wd" in df_cleaned.columns:
        wind_rose_plot(df_cleaned, "ws", "wd")

    histogram_and_distribution(df_cleaned, "ghi")
    histogram_and_distribution(df_cleaned, "ws")
    temperature_vs_humidity_analysis(df_cleaned)
    bubble_chart_ghi_vs_tamb(df_cleaned)

    print("\n=== EDA PIPELINE COMPLETE ===")
    return df_cleaned


if __name__ == "__main__":
    print("This module is intended to be imported, not executed directly.")
