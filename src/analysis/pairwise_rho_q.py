from itertools import combinations
from pathlib import Path

import pandas as pd

from src.analysis.rho_q import rho_q_by_scales


def run_pairwise_rho_q(
    returns_path,
    output_path,
    scales=None,
    q_values=None,
    poly_order=1,
):
    if scales is None:
        scales = [10, 20, 30, 40, 60, 80, 120, 160, 200, 250, 300]

    if q_values is None:
        q_values = [-4, -2, 0, 2, 4]

    returns_path = Path(returns_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(returns_path)

    # Remove date-like columns if present
    candidate_assets = [
        col for col in df.columns
        if col.lower() not in ["date", "datetime", "time"]
    ]

    results = []

    for asset_x, asset_y in combinations(candidate_assets, 2):
        pair_df = rho_q_by_scales(
            x=df[asset_x],
            y=df[asset_y],
            scales=scales,
            q_values=q_values,
            poly_order=poly_order,
        )

        pair_df.insert(0, "Asset_X", asset_x)
        pair_df.insert(1, "Asset_Y", asset_y)

        results.append(pair_df)

    final = pd.concat(results, ignore_index=True)
    final.to_csv(output_path, index=False)

    return final