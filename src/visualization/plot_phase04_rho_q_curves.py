from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_rho_q_curves(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df = df[df["q"] > 0].copy() ################

    df["Pair"] = df["Asset_X"] + " -- " + df["Asset_Y"]

    saved_files = []

    for pair, g in df.groupby("Pair"):
        plt.figure(figsize=(10, 6))

        for q, qg in g.groupby("q"):
            qg = qg.sort_values("scale")
            plt.plot(qg["scale"], qg["rho_q"], marker="o", linewidth=1, markersize=3, label=f"q={q}")

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("Scale n")
        plt.ylabel(r"$\rho_q(n)$")
        plt.title(f"Phase 4 Pairwise rho_q Curves\n{pair}")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()

        safe_name = (
            pair.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("--", "_")
        )

        output_path = output_dir / f"rho_q_curves_{safe_name}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        saved_files.append(str(output_path))

    return saved_files