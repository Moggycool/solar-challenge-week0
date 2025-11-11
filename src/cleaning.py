"""
cleaning.py
------------
Module for cleaning and preprocessing solar datasets.
"""


def clean_column_names(df):
    """
    Standardizes column names: lowercase, underscores, no spaces.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def remove_duplicates(df):
    """
    Removes duplicate rows, if any.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🧹 Removed {before - after} duplicate rows")
    return df


def handle_missing(df):
    """
    Fills missing numeric values with median and categorical with mode.
    """
    for col in df.columns:
        if df[col].dtype == 'O':  # object / categorical
            df[col].fillna(df[col].mode()[0] if not df[col].mode(
            ).empty else "Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def remove_outliers(df, cols, factor=1.5):
    """
    Removes outliers using the IQR (Interquartile Range) method.

    Args:
        df: pandas DataFrame
        cols: list of column names to check for outliers
        factor: IQR multiplier (default=1.5)
    """
    for col in cols:
        if df[col].dtype != 'O':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
