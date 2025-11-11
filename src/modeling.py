"""
modeling.py — Modeling module for solar dataset.

Provides reusable functions to train, evaluate, and save regression models
for predicting sensor outputs or irradiance.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_csv_safely, ensure_directory, print_section_header


# ---------------------------------------------------------------------
# 1️⃣ Dataset preparation
# ---------------------------------------------------------------------
def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split DataFrame into X (features) and y (target) train/test sets.
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    x = df[feature_cols]
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------
# 2️⃣ Regression model training
# ---------------------------------------------------------------------
def train_regression_model(
    model_type: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
):
    """
    Train a regression model.

    model_type options: "linear", "ridge", "lasso", "random_forest"
    """
    model_type = model_type.lower()

    if model_type == "linear":
        model = LinearRegression(**kwargs)
    elif model_type == "ridge":
        model = Ridge(**kwargs)
    elif model_type == "lasso":
        model = Lasso(**kwargs)
    elif model_type == "random_forest":
        model = RandomForestRegressor(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(x_train, y_train)
    return model


# ---------------------------------------------------------------------
# 3️⃣ Model evaluation
# ---------------------------------------------------------------------
def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate regression model using RMSE and R² score.
    """
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model Evaluation — RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return {"rmse": rmse, "r2": r2, "y_pred": y_pred}


def cross_validate_model(model, x: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """
    Perform cross-validation and return average R² score.
    """
    scores = cross_val_score(model, x, y, cv=cv, scoring="r2")
    mean_score = scores.mean()
    print(f"✅ Cross-validation mean R²: {mean_score:.3f}")
    return mean_score


# ---------------------------------------------------------------------
# 4️⃣ Save predictions
# ---------------------------------------------------------------------
def save_predictions(
    df_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    filename: str,
    folder: str = "predictions"
) -> None:
    """
    Save true vs predicted values as CSV.
    """
    ensure_directory(folder)
    df_out = df_test.copy()
    df_out["y_true"] = y_test.values
    df_out["y_pred"] = y_pred
    path = os.path.join(folder, filename)
    save_csv_safely(df_out, path)


# ---------------------------------------------------------------------
# 5️⃣ Full pipeline helper
# ---------------------------------------------------------------------
def run_model_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    model_type: str = "linear",
    test_size: float = 0.2,
    random_state: int = 42,
    filename: str = "predictions.csv"
) -> tuple:
    """
    Run full model pipeline: split, train, evaluate, cross-validate, save predictions.
    """
    print_section_header(f"Modeling Target: {target_col}")

    x_train, x_test, y_train, y_test = split_features_target(
        df, target_col, feature_cols, test_size, random_state
    )

    model = train_regression_model(model_type, x_train, y_train)
    eval_metrics = evaluate_model(model, x_test, y_test)
    cross_validate_model(model, x_train, y_train)

    save_predictions(x_test, y_test, eval_metrics["y_pred"], filename)
    return model, eval_metrics


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
