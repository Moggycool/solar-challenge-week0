"""
eda.py
------
Module for Exploratory Data Analysis (EDA) of solar datasets.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def correlation_heatmap(df, figsize=(10, 6)):
    """
    Displays a correlation heatmap for numerical variables.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def plot_distribution(df, column, bins=30):
    """
    Plots a histogram with KDE for a specific column.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_time_series(df, time_col, value_col):
    """
    Plots a simple time-series graph for solar measurements.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df[time_col], df[value_col], color='teal')
    plt.title(f"{value_col} over {time_col}")
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.grid(True)
    plt.show()


def compare_countries(dfs, column):
    """
    Compares the same variable across multiple country datasets.

    Args:
        dfs (dict): e.g., {"Togo": df_togo, "Benin": df_benin, "SierraLeone": df_sl}
        column (str): Column to compare
    """
    plt.figure(figsize=(10, 6))
    for name, df in dfs.items():
        sns.kdeplot(df[column], label=name)
    plt.title(f"Comparison of {column} across countries")
    plt.legend()
    plt.show()
