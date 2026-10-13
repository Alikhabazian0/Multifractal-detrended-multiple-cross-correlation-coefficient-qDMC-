import numpy as np
import pandas as pd


def validate_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    report = {
        "Asset": name,
        "Observations": len(df),
        "Start": df["date"].min(),
        "End": df["date"].max(),
        "Missing Dates": df["date"].isna().sum(),
        "Missing Prices": df["price"].isna().sum(),
        "Duplicate Dates": df["date"].duplicated().sum(),
        "Sorted": df["date"].is_monotonic_increasing,
        "Non-positive Prices": (df["price"] <= 0).sum(),
        "Minimum Price": df["price"].min(),
        "Maximum Price": df["price"].max(),
    }

    return pd.DataFrame([report])


def validate_matrix(df: pd.DataFrame, name: str) -> pd.DataFrame:
    report = {
        "Dataset": name,
        "Rows": len(df),
        "Columns": df.shape[1],
        "Start": df.index.min(),
        "End": df.index.max(),
        "Total Missing": df.isna().sum().sum(),
        "Total Infinite": np.isinf(df).sum().sum(),
        "Duplicate Dates": df.index.duplicated().sum(),
        "Sorted": df.index.is_monotonic_increasing,
    }

    return pd.DataFrame([report])