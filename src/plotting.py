from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def plot_returns_timeseries(returns: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean = returns.replace([np.inf, -np.inf], np.nan)

    n_assets = len(clean.columns)
    fig, axes = plt.subplots(
        n_assets,
        1,
        figsize=(12, 2.6 * n_assets),
        sharex=True,
    )

    if n_assets == 1:
        axes = [axes]

    for ax, col in zip(axes, clean.columns):
        ax.plot(clean.index, clean[col], linewidth=0.8)
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"Log Returns: {col}")
        ax.set_ylabel("Return")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qq_plots(returns: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean = returns.replace([np.inf, -np.inf], np.nan)

    n_assets = len(clean.columns)
    n_cols = 2
    n_rows = int(np.ceil(n_assets / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 4 * n_rows),
    )

    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, clean.columns):
        x = pd.to_numeric(clean[col], errors="coerce").dropna()
        stats.probplot(x, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {col}")

    for ax in axes[n_assets:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)