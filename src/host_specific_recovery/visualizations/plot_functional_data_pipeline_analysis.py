import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
import networkx as nx


def plot_two_hists_with_auc(A, B, bins="fd", n_bins: int | None = None, density=True, alpha=0.35,
                            colors=("#ff6a3a", "#a62a0d"), labels=("A", "B"),
                            xlabel="Mean Weighted Jaccard similarity", ylabel="Density", x_fontsize=20,
                            tick_fontsize=16, path: str | None = None, dpi: int = 300, legend_fontsize: int = 12):
    """
    :param A: Array-like of values
    :param B: Array-like of values
    :param bins: Method or number of bins for histogram (see numpy.histogram_bin_edges)
    :param n_bins: If specified, overrides 'bins' parameter to use this exact number of bins
    :param density: If True, plot probability density instead of counts
    :param alpha: Transparency of histogram bars
    :param colors: Colors for the two histograms
    :param labels: Labels for the two histograms
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param x_fontsize: font size for x and y labels
    :param tick_fontsize: font size for x ticks
    :param path: Path to save the figure (if None, figure is not saved)
    :param dpi: DPI for saving the figure
    :param legend_fontsize: Font size for the legend
    :return: Matplotlib Axes object
    """
    A = np.asarray(A, float).ravel()
    B = np.asarray(B, float).ravel()
    A = A[np.isfinite(A)]
    B = B[np.isfinite(B)]

    # AUC
    scores = np.concatenate([A, B])
    y_true = np.concatenate([np.ones(A.size, dtype=int), np.zeros(B.size, dtype=int)])
    auc = float(roc_auc_score(y_true, scores))

    # bins
    data_min = float(np.min([A.min(), B.min()]))
    data_max = float(np.max([A.max(), B.max()]))

    if n_bins is not None:
        edges = np.linspace(data_min, data_max, int(n_bins) + 1)
    else:
        if bins is None:
            bins = "fd"
        if isinstance(bins, int):
            edges = np.linspace(data_min, data_max, int(bins) + 1)
        else:
            edges = np.histogram_bin_edges(np.concatenate([A, B]), bins=bins, range=(data_min, data_max))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.hist(A, bins=edges, density=density, alpha=alpha,
            color=colors[0], label=labels[0])
    ax.hist(B, bins=edges, density=density, alpha=alpha,
            color=colors[1], label=labels[1])

    ax.set_xlabel(xlabel, fontsize=x_fontsize)
    ax.set_ylabel(ylabel, fontsize=x_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(0.05))

    ax.set_xlim(data_min, data_max)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))

    ax.tick_params(axis="y", left=False, labelleft=False)

    ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.90), ncol=1, frameon=False, fontsize=legend_fontsize)

    ax.text(0.5, 0.98, f"AUC = {auc:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=x_fontsize)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    ax.tick_params(axis="both", width=2.0)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    return ax

def plot_similarity_network(S_AC: pd.DataFrame, S_BC: pd.DataFrame, S_AD: pd.DataFrame, S_BD: pd.DataFrame,
                            threshold_quantile: float = 0.9, node_size: int = 3000, figsize=(9, 13), A_color="#ff6a3a",
                            B_color="#a62a0d", C_color="#66aa00", D_color="#1f77b4", auto_y: bool = True,
                            gap: float = 0.02, gap_AB: float = 0.1, gap_CD: float = 0.0,  shuffle: bool = True,
                            seed: int | None = 1, path: str | None = None,):

    if not S_AC.columns.equals(S_BC.columns):
        raise ValueError("S_AC.columns and S_BC.columns must be identical (same C taxa, same order).")
    if not S_AD.columns.equals(S_BD.columns):
        raise ValueError("S_AD.columns and S_BD.columns must be identical (same D taxa, same order).")

    if not (0.0 <= gap < 1.0):
        raise ValueError("gap must be in [0, 1).")
    if not (0.0 <= gap_AB < 1.0):
        raise ValueError("gap_AB must be in [0, 1).")
    if not (0.0 <= gap_CD < 1.0):
        raise ValueError("gap_CD must be in [0, 1).")
    if not auto_y:
        raise ValueError("auto_y=False not supported in this version.")

    A = list(map(str, S_AC.index))
    B = list(map(str, S_BC.index))
    C = list(map(str, S_AC.columns))
    D = list(map(str, S_AD.columns))

    rng = np.random.default_rng(seed) if shuffle else None
    if shuffle:
        rng.shuffle(A)
        rng.shuffle(B)
        rng.shuffle(C)
        rng.shuffle(D)

    S_AC = S_AC.copy()
    S_BC = S_BC.copy()
    S_AD = S_AD.copy()
    S_BD = S_BD.copy()

    for M in (S_AC, S_BC, S_AD, S_BD):
        M.index = list(map(str, M.index))
        M.columns = list(map(str, M.columns))

    S_AC = S_AC.loc[A, C]
    S_BC = S_BC.loc[B, C]
    S_AD = S_AD.loc[A, D]
    S_BD = S_BD.loc[B, D]

    nA, nB, nC, nD = len(A), len(B), len(C), len(D)
    if (nA + nB) == 0:
        raise ValueError("Groups A and/or B must be non-empty.")
    if (nC + nD) == 0:
        raise ValueError("Groups C and/or D must be non-empty.")

    q = float(threshold_quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError("threshold_quantile must be in [0, 1].")

    mats = [
        S_AC.to_numpy(dtype=float, copy=False).ravel(),
        S_BC.to_numpy(dtype=float, copy=False).ravel(),
        S_AD.to_numpy(dtype=float, copy=False).ravel(),
        S_BD.to_numpy(dtype=float, copy=False).ravel(),
    ]
    vals = np.concatenate(mats)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite similarity values found.")
    threshold = float(np.quantile(vals, q))

    G = nx.Graph()
    A_nodes = [f"A:{t}" for t in A]
    B_nodes = [f"B:{t}" for t in B]
    C_nodes = [f"C:{t}" for t in C]
    D_nodes = [f"D:{t}" for t in D]

    G.add_nodes_from(A_nodes, part="A")
    G.add_nodes_from(B_nodes, part="B")
    G.add_nodes_from(C_nodes, part="C")
    G.add_nodes_from(D_nodes, part="D")

    edges = []
    for i, a in enumerate(A):
        for j, c in enumerate(C):
            w = float(S_AC.iat[i, j])
            if np.isfinite(w) and w >= threshold:
                edges.append((f"A:{a}", f"C:{c}", w))
        for j, d in enumerate(D):
            w = float(S_AD.iat[i, j])
            if np.isfinite(w) and w >= threshold:
                edges.append((f"A:{a}", f"D:{d}", w))

    for i, b in enumerate(B):
        for j, c in enumerate(C):
            w = float(S_BC.iat[i, j])
            if np.isfinite(w) and w >= threshold:
                edges.append((f"B:{b}", f"C:{c}", w))
        for j, d in enumerate(D):
            w = float(S_BD.iat[i, j])
            if np.isfinite(w) and w >= threshold:
                edges.append((f"B:{b}", f"D:{d}", w))

    edges.sort(key=lambda t: t[2], reverse=True)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    def linspace_positions(n, y0, y1):
        if n <= 1:
            return np.array([(y0 + y1) / 2.0])
        return np.linspace(y0, y1, n)

    top_margin = gap / 2.0
    bot_margin = gap / 2.0
    avail = 1.0 - gap

    denom_right = max((nA - 1), 0) + max((nB - 1), 0)
    if denom_right == 0:
        s_max_right = float("inf")
    else:
        s_max_right = max((avail - gap_AB) / float(denom_right), 0.0)

    denom_left = max((nC - 1), 0) + max((nD - 1), 0)
    if denom_left == 0:
        s_max_left = float("inf")
    else:
        s_max_left = max((avail - gap_CD) / float(denom_left), 0.0)

    s = min(s_max_left, s_max_right)
    if not np.isfinite(s):
        s = 0.0
    if s <= 0.0 and (max(nA, nB, nC, nD) > 1):
        raise ValueError("Not enough vertical space to place nodes with the requested gap_AB/gap_CD/gap.")

    hA = s * max(nA - 1, 0)
    hB = s * max(nB - 1, 0)
    hC = s * max(nC - 1, 0)
    hD = s * max(nD - 1, 0)

    hAB_total = hA + gap_AB + hB
    AB_lo = max(0.5 - hAB_total / 2.0, bot_margin)
    AB_hi = min(0.5 + hAB_total / 2.0, 1.0 - top_margin)
    if (AB_hi - AB_lo) + 1e-12 < hAB_total:
        raise ValueError("Right column cannot fit A+B+gap_AB within margins. Reduce gap_AB or gap.")
    A_y = (AB_lo, AB_lo + hA)
    B_y = (AB_hi - hB, AB_hi)

    hCD_total = hD + gap_CD + hC
    CD_lo = max(0.5 - hCD_total / 2.0, bot_margin)
    CD_hi = min(0.5 + hCD_total / 2.0, 1.0 - top_margin)
    if (CD_hi - CD_lo) + 1e-12 < hCD_total:
        raise ValueError("Left column cannot fit D+C+gap_CD within margins. Reduce gap_CD or gap.")
    D_y = (CD_lo, CD_lo + hD)
    C_y = (CD_hi - hC, CD_hi)

    pos = {}
    x_left, x_right = 0.0, 2.0

    for node, y in zip(D_nodes, linspace_positions(len(D_nodes), *D_y)):
        pos[node] = (x_left, float(y))
    for node, y in zip(C_nodes, linspace_positions(len(C_nodes), *C_y)):
        pos[node] = (x_left, float(y))

    for node, y in zip(A_nodes, linspace_positions(len(A_nodes), *A_y)):
        pos[node] = (x_right, float(y))
    for node, y in zip(B_nodes, linspace_positions(len(B_nodes), *B_y)):
        pos[node] = (x_right, float(y))

    wts = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    if wts.size:
        w_min, w_max = float(wts.min()), float(wts.max())
        denom = (w_max - w_min) if (w_max > w_min) else 1.0
        widths = 0.2 + 1.2 * (wts - w_min) / denom
        alphas = 0.10 + 0.70 * (wts - w_min) / denom
    else:
        widths, alphas = [], []

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos, nodelist=A_nodes, node_size=node_size, node_color=A_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=B_nodes, node_size=node_size, node_color=B_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=C_nodes, node_size=node_size, node_color=C_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=D_nodes, node_size=node_size, node_color=D_color, ax=ax)

    for (u, v), lw, a in zip(G.edges(), widths, alphas):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a), ax=ax)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax

def plot_effectsize_vs_effectsize(eff1, p1, eff2, p2, alpha=0.05, xlabel="Effect size (Group 1)",
                                  ylabel="Effect size (Group 2)", title=None, s=60, xlim=(0.5, 1.0),
                                  ylim=(0.5, 1.0), eq_line=True, groups=("Group 1", "Group 2"),
                                  path=None, dpi=300, fontsize=16, ticksize=14, fontsize_leg=12):

    eff1 = np.asarray(eff1, dtype=float).ravel()
    p1 = np.asarray(p1, dtype=float).ravel()
    eff2 = np.asarray(eff2, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    if not (eff1.size == p1.size == eff2.size == p2.size):
        raise ValueError("Inputs must have the same length (or set filter_input=True).")

    sig1 = p1 < alpha
    sig2 = p2 < alpha

    both = sig1 & sig2
    only1 = sig1 & ~sig2
    only2 = ~sig1 & sig2
    none = ~sig1 & ~sig2

    fig, ax = plt.subplots(figsize=(6, 6))

    if np.sum(none) > 0:
        ax.scatter(eff1[none],  eff2[none],  marker="o", s=s, alpha=0.75, label="Not significant")
    else:
        ax.scatter(eff1[none], eff2[none], marker="o", s=s, alpha=0.75, label="_nolegend_")
    if np.sum(only1) > 0:
        ax.scatter(eff1[only1], eff2[only1], marker="^", s=s, alpha=0.75, label=f"Only {groups[0]} significant")
    else:
        ax.scatter(eff1[only1], eff2[only1], marker="^", s=s, alpha=0.75, label="_nolegend_")
    if np.sum(only2) > 0:
        ax.scatter(eff1[only2], eff2[only2], marker="s", s=s, alpha=0.75, label=f"only {groups[1]} significant")
    else:
        ax.scatter(eff1[only2], eff2[only2], marker="s", s=s, alpha=0.75, label="_nolegend_")
    if np.sum(both) > 0:
        ax.scatter(eff1[both],  eff2[both],  marker="D", s=s, alpha=0.75, label="Both significant")
    else:
        ax.scatter(eff1[both], eff2[both], marker="D", s=s, alpha=0.75, label="_nolegend_")

    if eq_line:
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k--", lw=1.5, label="_nolegend_")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=fontsize_leg, loc="upper left")
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig, ax
