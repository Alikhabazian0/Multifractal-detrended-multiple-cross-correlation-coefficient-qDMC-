from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def short_name(name):
    name = name.strip()

    if "Net Zero" in name:
        return "ESG"
    if "SDG" in name:
        return "SDG"
    if "Clean Energy" in name:
        return "Clean Energy"
    if "Artificial Intelligence" in name or "Kensho" in name:
        return "AI"

    return name


def edge_color(weight):
    if weight >= 0.90:
        return "darkgreen"
    elif weight >= 0.70:
        return "green"
    elif weight >= 0.50:
        return "orange"
    else:
        return "red"


def plot_phase04_correlation_network(
    input_path,
    output_png,
    output_pdf,
    output_table,
    q_value=2,
):
    input_path = Path(input_path)
    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_table = Path(output_table)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    df = df[df["q"] >0 ].copy()    ############################

    df_q = df[df["q"] == q_value].copy()

    if df_q.empty:
        raise ValueError(f"No rows found for q = {q_value}")

    df_q["Asset_X_short"] = df_q["Asset_X"].apply(short_name)
    df_q["Asset_Y_short"] = df_q["Asset_Y"].apply(short_name)

    edge_table = (
        df_q.groupby(["Asset_X_short", "Asset_Y_short"])
        .agg(Mean_rho_q=("rho_q", "mean"))
        .reset_index()
    )

    edge_table["Rank"] = (
        edge_table["Mean_rho_q"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    edge_table = edge_table.sort_values("Mean_rho_q", ascending=False)
    edge_table.to_csv(output_table, index=False)

    G = nx.Graph()

    node_colors = {
        "ESG": "steelblue",
        "SDG": "forestgreen",
        "Clean Energy": "orange",
        "AI": "mediumpurple",
    }

    for _, row in edge_table.iterrows():
        x = row["Asset_X_short"]
        y = row["Asset_Y_short"]
        w = row["Mean_rho_q"]

        G.add_edge(x, y, weight=w)

    pos = nx.kamada_kawai_layout(G, weight="weight")

    plt.figure(figsize=(8, 6))

    nodes = list(G.nodes())
    colors = [node_colors.get(n, "lightgray") for n in nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=2600,
        edgecolors="black",
        linewidths=1.2,
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [2 + 8 * w for w in weights]
    colors_edges = [edge_color(w) for w in weights]

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color=colors_edges,
        alpha=0.75,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=11,
        font_weight="bold",
    )

    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.3f}"
        for u, v in G.edges()
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        label_pos=0.5,
    )

    plt.title(
        rf"Phase 4 Correlation Network: Mean $\rho_q(n)$ at $q={q_value}$",
        fontsize=13,
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    return edge_table