from pathlib import Path
import pandas as pd


def read_sp_excel(path: str | Path, sheet_name: str = "Performance Graph") -> pd.DataFrame:
    """
    Read raw S&P index Excel export.

    Raw structure verified:
    - rows 0–5: metadata
    - row 6: header
    - rows after final observation: NaN/disclaimer footer
    """
    path = Path(path)

    return pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=6,
        engine="xlrd",
    )