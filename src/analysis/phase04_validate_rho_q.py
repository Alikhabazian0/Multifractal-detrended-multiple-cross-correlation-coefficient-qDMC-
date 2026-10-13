from pathlib import Path
import pandas as pd
import numpy as np


def validate_phase04_rho_q(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)


    # Exclude q=0 until q=0 formula is audited
    df = df[df["q"] > 0].copy()      # added new to exclude q=0


    required = {"Asset_X", "Asset_Y", "q", "scale", "rho_q"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df["Pair"] = df["Asset_X"] + " -- " + df["Asset_Y"]

    rows = []

    for pair, g in df.groupby("Pair"):
        values = g["rho_q"]

        negative_q = g[g["q"] < 0]
        if len(negative_q) == 0:
            n_negative_q_out_of_bounds = 0
            pass_negative_q_bounds = True
        else:
            n_negative_q_out_of_bounds = (
                negative_q["rho_q"].abs() > 1.0 + 1e-12
            ).sum()
            pass_negative_q_bounds = n_negative_q_out_of_bounds == 0

        rows.append({
            "Pair": pair,
            "N_Rows": len(g),
            "N_q": g["q"].nunique(),
            "N_scales": g["scale"].nunique(),
            "Min_rho": values.min(),
            "Max_rho": values.max(),
            "Mean_rho": values.mean(),
            "Std_rho": values.std(),
            "N_NaN": values.isna().sum(),
            "N_Negative_q_Out_Of_Bounds": n_negative_q_out_of_bounds,
            "Scale_Variation_Mean": g.groupby("q")["rho_q"].std().mean(),
            "Q_Variation_Mean": g.groupby("scale")["rho_q"].std().mean(),
            "Pass_Negative_q_Bounds": pass_negative_q_bounds,
        })

    report = pd.DataFrame(rows)

    report["Pass_No_NaN"] = report["N_NaN"] == 0
    report["Pass_Scale_Variation"] = report["Scale_Variation_Mean"] > 1e-6
    report["Pass_Q_Variation"] = report["Q_Variation_Mean"] > 1e-6

    report["Overall_Pass"] = (
        report["Pass_Negative_q_Bounds"]
        & report["Pass_No_NaN"]
        & report["Pass_Scale_Variation"]
        & report["Pass_Q_Variation"]
    )

    report.to_csv(output_path, index=False)

    return report