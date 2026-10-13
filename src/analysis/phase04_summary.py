from pathlib import Path
import pandas as pd


def classify_strength(mean_rho):
    if mean_rho >= 0.90:
        return "Very Strong"
    elif mean_rho >= 0.70:
        return "Strong"
    elif mean_rho >= 0.50:
        return "Moderate"
    elif mean_rho >= 0.30:
        return "Weak"
    else:
        return "Very Weak"


def summarize_phase04(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Exclude q=0 until q=0 implementation is resolved
    df = df[df["q"] > 0].copy()     # added new to exclude q=0 from summary

    df["Pair"] = df["Asset_X"] + " -- " + df["Asset_Y"]

    summary = (
        df.groupby("Pair")
        .agg(
            Mean_rho=("rho_q", "mean"),
            Median_rho=("rho_q", "median"),
            Std_rho=("rho_q", "std"),
            Min_rho=("rho_q", "min"),
            Max_rho=("rho_q", "max"),
            Mean_q=("q", "mean"),
            N_q=("q", "nunique"),
            N_scales=("scale", "nunique"),
        )
        .reset_index()
    )

    # Range of rho_q across all scales and q values
    summary["Delta_rho"] = (
        summary["Max_rho"] - summary["Min_rho"]
    )
    
    # Author-defined interpretation (not from the qDMC paper)
    summary["Author_Interpretation"] = (
        summary["Mean_rho"]
        .apply(classify_strength)
    )

    summary = summary[
    [
        "Pair",
        "Mean_rho",
        "Median_rho",
        "Std_rho",
        "Min_rho",
        "Max_rho",
        "Delta_rho",
        "Mean_q",
        "N_q",
        "N_scales",
        "Author_Interpretation",
    ]
]

    summary = summary.sort_values(
    "Mean_rho",
    ascending=False,
)

    summary.to_csv(output_path, index=False)

    return summary