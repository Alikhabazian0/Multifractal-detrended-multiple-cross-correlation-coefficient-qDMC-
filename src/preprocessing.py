from pathlib import Path
import numpy as np
import pandas as pd

from src.data_io import read_sp_excel


def load_sp_index_excel(path: str | Path) -> pd.DataFrame:
    raw = read_sp_excel(path)

    raw.columns = raw.columns.str.strip()

    date_col = raw.columns[0]
    price_col = raw.columns[1]

    df = raw[[date_col, price_col]].copy()
    df.columns = ["date", "price"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["date", "price"])
    out = out.sort_values("date")
    out = out.drop_duplicates(subset="date", keep="last")
    out = out.reset_index(drop=True)
    return out


def align_indices(clean_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []

    for name, df in clean_data.items():
        s = df.set_index("date")["price"].rename(name)
        series.append(s)

    price_matrix = pd.concat(series, axis=1).sort_index()
    aligned_prices = price_matrix.dropna(how="any")

    return aligned_prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if (prices <= 0).any().any():
        bad_cols = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"Non-positive prices found in: {bad_cols}")

    returns = np.log(prices).diff()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return returns