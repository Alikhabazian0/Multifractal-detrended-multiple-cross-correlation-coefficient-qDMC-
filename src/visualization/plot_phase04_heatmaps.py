from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_phase04_heatmaps(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df = df[df["q"] > 0].copy() #####################

    df["Pair"] = df["Asset_X"] + " -- " + df["Asset_Y"]

    saved_files = []

    for pair, g in df.groupby("Pair"):
        pivot = (
            g.pivot_table(
                index="q",
                columns="scale",
                values="rho_q",
                aggfunc="mean",
            )
            .sort_index(ascending=False)
        )

        plt.figure(figsize=(11, 5.5))

        im = plt.imshow(
            pivot.values,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )

        plt.colorbar(im, label=r"$\rho_q(n)$")

        plt.xticks(
            ticks=range(len(pivot.columns)),
            labels=pivot.columns,
            rotation=90,
            fontsize=7,
        )

        plt.yticks(
            ticks=range(len(pivot.index)),
            labels=pivot.index,
        )

        plt.xlabel("Scale n")
        plt.ylabel("q")
        plt.title(f"Phase 4 Heatmap of $\\rho_q(n)$\n{pair}")

        plt.tight_layout()

        safe_name = (
            pair.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("--", "_")
        )

        output_path = output_dir / f"heatmap_rho_q_{safe_name}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        saved_files.append(str(output_path))

    return saved_files