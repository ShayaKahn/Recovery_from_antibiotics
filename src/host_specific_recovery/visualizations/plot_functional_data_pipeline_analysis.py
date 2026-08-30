import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
import networkx as nx


def plot_functional_redundancy_over_time(fun_redundancy, sample_labels=None, timepoint_labels=None,
                                         path: str | None = None, dpi: int = 300, figsize=(6, 5),
                                         subject_color="0.75",
                                         mean_color="black", subject_alpha=0.55, subject_lw=1.0,
                                         mean_lw=3.0, fontsize=16, ticksize=12):
    """
    Plot functional redundancy trajectories for all subjects and their mean.

    :param fun_redundancy: Subject-by-time array-like functional redundancy values.
    :param sample_labels: Optional labels for the time axis.
    :param timepoint_labels: Optional labels for the time axis. Preferred alias for sample_labels.
    :param path: Path to save the figure. If None, display the figure.
    :param dpi: DPI for saving the figure.
    :param figsize: Figure size.
    :param subject_color: Color for individual subject trajectories.
    :param mean_color: Color for the mean trajectory.
    :param subject_alpha: Alpha for individual subject trajectories.
    :param subject_lw: Line width for individual subject trajectories.
    :param mean_lw: Line width for mean trajectory.
    :param fontsize: Axis label font size.
    :param ticksize: Tick label font size.
    :return: Matplotlib Axes object.
    """
    fun_redundancy = np.asarray(fun_redundancy, dtype=float)
    if fun_redundancy.ndim == 1:
        fun_redundancy = fun_redundancy[:, np.newaxis]
    if fun_redundancy.ndim != 2:
        raise ValueError("fun_redundancy must be a 1D or 2D array-like object.")
    if fun_redundancy.shape[0] == 0 or fun_redundancy.shape[1] == 0:
        raise ValueError("fun_redundancy must contain at least one subject and one time point.")

    n_time = fun_redundancy.shape[1]
    if timepoint_labels is not None:
        if sample_labels is not None:
            raise ValueError("Use either sample_labels or timepoint_labels, not both.")
        sample_labels = timepoint_labels
    if sample_labels is None:
        sample_labels = [str(i) for i in range(n_time)]
    if len(sample_labels) != n_time:
        raise ValueError(f"sample_labels has {len(sample_labels)} labels for {n_time} time points.")

    x = np.arange(n_time)
    mean_redundancy = np.nanmean(fun_redundancy, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    for subject_values in fun_redundancy:
        ax.plot(x, subject_values, color=subject_color, alpha=subject_alpha, lw=subject_lw)
    ax.plot(x, mean_redundancy, color=mean_color, lw=mean_lw, label="Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels, rotation=45, ha="right")
    ax.set_ylabel("Functional redundancy", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    #ax.legend(frameon=False, fontsize=ticksize)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    return ax


def plot_functional_specificity_over_time(functional_specificity, sample_labels=None, timepoint_labels=None,
                                          path: str | None = None, dpi: int = 300, figsize=(6, 5),
                                          subject_color="0.75",
                                          mean_color="black", subject_alpha=0.55, subject_lw=1.0,
                                          mean_lw=3.0, fontsize=16, ticksize=12, ylim=(0.0, 1.0),
                                          y_min: float | None = None):
    """
    Plot functional specificity trajectories for all subjects and their mean.

    :param functional_specificity: Subject-by-time array-like specificity values.
    :param sample_labels: Optional labels for the time axis.
    :param timepoint_labels: Optional labels for the time axis. Preferred alias for sample_labels.
    :param path: Path to save the figure. If None, display the figure.
    :param dpi: DPI for saving the figure.
    :param figsize: Figure size.
    :param subject_color: Color for individual subject trajectories.
    :param mean_color: Color for the mean trajectory.
    :param subject_alpha: Alpha for individual subject trajectories.
    :param subject_lw: Line width for individual subject trajectories.
    :param mean_lw: Line width for mean trajectory.
    :param fontsize: Axis label font size.
    :param ticksize: Tick label font size.
    :param ylim: Optional y-axis limits. Set to None to use Matplotlib defaults.
    :param y_min: Optional y-axis lower limit. Overrides the lower value in ylim.
    :return: Matplotlib Axes object.
    """
    functional_specificity = np.asarray(functional_specificity, dtype=float)
    if functional_specificity.ndim == 1:
        functional_specificity = functional_specificity[:, np.newaxis]
    if functional_specificity.ndim != 2:
        raise ValueError("functional_specificity must be a 1D or 2D array-like object.")
    if functional_specificity.shape[0] == 0 or functional_specificity.shape[1] == 0:
        raise ValueError("functional_specificity must contain at least one subject and one time point.")

    n_time = functional_specificity.shape[1]
    if timepoint_labels is not None:
        if sample_labels is not None:
            raise ValueError("Use either sample_labels or timepoint_labels, not both.")
        sample_labels = timepoint_labels
    if sample_labels is None:
        sample_labels = [str(i) for i in range(n_time)]
    if len(sample_labels) != n_time:
        raise ValueError(f"sample_labels has {len(sample_labels)} labels for {n_time} time points.")

    x = np.arange(n_time)
    mean_specificity = np.nanmean(functional_specificity, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    for subject_values in functional_specificity:
        ax.plot(x, subject_values, color=subject_color, alpha=subject_alpha, lw=subject_lw)
    ax.plot(x, mean_specificity, color=mean_color, lw=mean_lw, label="Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(sample_labels, rotation=45, ha="right")
    ax.set_ylabel("Functional specificity", fontsize=fontsize)
    if ylim is not None:
        y_low, y_high = ylim
        if y_min is not None:
            y_low = y_min
        ax.set_ylim(y_low, y_high)
    elif y_min is not None:
        ax.set_ylim(bottom=y_min)
    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    return ax


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
    else:
        plt.show()


def plot_similarity_network(S_AC: pd.DataFrame, S_BC: pd.DataFrame, S_AD: pd.DataFrame, S_BD: pd.DataFrame,
                            threshold_quantile: float = 0.9, node_size: int = 3000, figsize=(9, 13),
                            A_color="#ff6a3a", B_color="#a62a0d", C_color="#66aa00",
                            D_color="#1f77b4", auto_y: bool = True, gap: float = 0.02, gap_AB: float = 0.1,
                            gap_CD: float = 0.0,  shuffle: bool = True, seed: int | None = 1,
                            path: str | None = None, hide_set: str | None = None):

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
    valid_hide_sets = {None, "A", "B", "C", "D"}
    if hide_set not in valid_hide_sets:
        raise ValueError("hide_set must be one of None, 'A', 'B', 'C', or 'D'.")

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

    hide_parts = {
        None: set(),
        "A": {"A"},
        "B": {"B"},
        "C": {"C"},
        "D": {"D"},
    }[hide_set]
    hidden_nodes = {node for part in hide_parts for node in {
        "A": A_nodes,
        "B": B_nodes,
        "C": C_nodes,
        "D": D_nodes,
    }[part]}

    visible_A_nodes = [node for node in A_nodes if node not in hidden_nodes]
    visible_B_nodes = [node for node in B_nodes if node not in hidden_nodes]
    visible_C_nodes = [node for node in C_nodes if node not in hidden_nodes]
    visible_D_nodes = [node for node in D_nodes if node not in hidden_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=visible_A_nodes, node_size=node_size, node_color=A_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visible_B_nodes, node_size=node_size, node_color=B_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visible_C_nodes, node_size=node_size, node_color=C_color, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visible_D_nodes, node_size=node_size, node_color=D_color, ax=ax)

    for (u, v), lw, a in zip(G.edges(), widths, alphas):
        if u in hidden_nodes or v in hidden_nodes:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=float(lw), alpha=float(a), ax=ax)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


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
    else:
        plt.show()
