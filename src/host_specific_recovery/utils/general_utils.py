import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

def subset(post_matrix, base_sample, ABX_sample, strict, new):
    num_timepoints = post_matrix.shape[0]
    op_lst = [operator.ne] * num_timepoints
    if new:
        general_cond = [base_sample == 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    else:
        general_cond = [base_sample != 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    timepoints_vals = []
    if strict:
        for j in range(num_timepoints):
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            special_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1, num_timepoints)]
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                op_lst[j] = operator.eq
    else:
        # iterate over the time points.
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            # combine the conditions.
            cond = general_cond + inter_cond
            # find the returned species at each time point.
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
                op_lst[j] = operator.eq
    return timepoints_vals

def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    return pvals_adj

def p_to_label(p, z):
    if p <= 0.001:
        stars = '***'
    elif p <= 0.01:
        stars = '**'
    elif p <= 0.05:
        stars = '*'
    else:
        return 'ns'

    symbol = '*' if z > 0 else '#'

    return '\n'.join(symbol for _ in stars)

def p_to_label_one_sided(p):
    if p <= 0.001:
        stars = '***'
    elif p <= 0.01:
        stars = '**'
    elif p <= 0.05:
        stars = '*'
    else:
        return 'ns'
    return '\n'.join(list(stars))


def plot_pcoa_and_distance_heatmap(data, metric='jaccard', figsize=(12, 5), cmap='viridis',
                                   pcoa_ax=None, heatmap_ax=None, sample_colors=None,
                                   title_prefix=None):
    """
    Calculate and plot PCoA plus a sample-by-sample distance heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature table where rows are features/taxa and columns are samples.
    metric : str, default 'jaccard'
        Distance metric. Supported values are 'jaccard', 'braycurtis', and
        'root_jensen_shannon' (aliases: 'rjsd', 'jensenshannon').
    figsize : tuple, default (12, 5)
        Figure size used when axes are not provided.
    cmap : str, default 'viridis'
        Matplotlib colormap for the heatmap.
    pcoa_ax : matplotlib.axes.Axes or None
        Existing axis for the PCoA plot. If None, a new figure is created.
    heatmap_ax : matplotlib.axes.Axes or None
        Existing axis for the heatmap. If None, a new figure is created.
    sample_colors : array-like or None
        Optional colors for samples in the PCoA scatter plot.
    title_prefix : str or None
        Optional prefix for plot titles.

    Returns
    -------
    dict
        Contains the figure, axes, distance matrix dataframe, skbio distance
        matrix, and PCoA results.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with samples in columns")
    if data.shape[1] < 2:
        raise ValueError("data must contain at least two sample columns")
    if (pcoa_ax is None) != (heatmap_ax is None):
        raise ValueError("provide both pcoa_ax and heatmap_ax, or provide neither")

    from skbio import DistanceMatrix
    from skbio.stats.ordination import pcoa

    metric_key = metric.lower().replace("-", "_").replace(" ", "_")
    metric_aliases = {
        'jaccard': 'jaccard',
        'braycurtis': 'braycurtis',
        'bray_curtis': 'braycurtis',
        'root_jensen_shannon': 'jensenshannon',
        'root_jensen_shannon_divergence': 'jensenshannon',
        'rjsd': 'jensenshannon',
        'jensenshannon': 'jensenshannon',
    }

    if metric_key not in metric_aliases:
        raise ValueError(
            "metric must be one of: 'jaccard', 'braycurtis', or 'root_jensen_shannon'"
        )

    scipy_metric = metric_aliases[metric_key]
    sample_ids = data.columns.astype(str).tolist()
    sample_matrix = data.to_numpy(dtype=float).T

    if np.any(~np.isfinite(sample_matrix)):
        raise ValueError("data contains NaN or infinite values")
    if scipy_metric in {'braycurtis', 'jensenshannon'} and np.any(sample_matrix < 0):
        raise ValueError(f"{metric} distance requires non-negative values")

    if scipy_metric == 'jaccard':
        distance_input = sample_matrix > 0
    else:
        distance_input = sample_matrix

    distances = squareform(pdist(distance_input, metric=scipy_metric))
    distance_df = pd.DataFrame(distances, index=sample_ids, columns=sample_ids)
    distance_matrix = DistanceMatrix(distance_df.to_numpy(), ids=sample_ids)
    ordination = pcoa(distance_matrix)

    if pcoa_ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        pcoa_ax = axes[0]
        heatmap_ax = axes[1]
    else:
        fig = pcoa_ax.figure

    title_metric = {
        'jaccard': 'Jaccard',
        'braycurtis': 'Bray-Curtis',
        'jensenshannon': 'Root Jensen-Shannon',
    }[scipy_metric]
    prefix = f"{title_prefix} - " if title_prefix else ""

    coords = ordination.samples.iloc[:, :2]
    var_exp = ordination.proportion_explained.iloc[:2] * 100

    pcoa_ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=sample_colors, s=70, edgecolor='black')
    for sample_id, x, y in zip(sample_ids, coords.iloc[:, 0], coords.iloc[:, 1]):
        pcoa_ax.text(x, y, sample_id, fontsize=8, ha='left', va='bottom')
    pcoa_ax.axhline(0, color='lightgray', linewidth=0.8, zorder=0)
    pcoa_ax.axvline(0, color='lightgray', linewidth=0.8, zorder=0)
    pcoa_ax.set_xlabel(f"PCoA1 ({var_exp.iloc[0]:.1f}%)")
    pcoa_ax.set_ylabel(f"PCoA2 ({var_exp.iloc[1]:.1f}%)")
    pcoa_ax.set_title(f"{prefix}PCoA ({title_metric} distance)")

    heatmap_values = distance_df.to_numpy().copy()
    diagonal_mask = np.eye(heatmap_values.shape[0], dtype=bool)
    off_diagonal = heatmap_values[~diagonal_mask]
    vmin = np.min(off_diagonal) if off_diagonal.size else 0
    vmax = np.max(off_diagonal) if off_diagonal.size else 1

    mean_distances = np.ma.array(heatmap_values, mask=diagonal_mask).mean(axis=1).filled(0)
    heatmap_order = np.argsort(-mean_distances)
    heatmap_values_sorted = heatmap_values[np.ix_(heatmap_order, heatmap_order)]
    sample_ids_sorted = [sample_ids[i] for i in heatmap_order]
    heatmap_masked = np.ma.array(
        heatmap_values_sorted,
        mask=np.eye(heatmap_values_sorted.shape[0], dtype=bool)
    )

    image = heatmap_ax.imshow(heatmap_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    heatmap_ax.set_xticks(np.arange(len(sample_ids_sorted)))
    heatmap_ax.set_yticks(np.arange(len(sample_ids_sorted)))
    heatmap_ax.set_xticklabels(sample_ids_sorted, rotation=90)
    heatmap_ax.set_yticklabels(sample_ids_sorted)
    heatmap_ax.set_title(f"{prefix}{title_metric} distance matrix")
    fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    return {
        'figure': fig,
        'pcoa_axis': pcoa_ax,
        'heatmap_axis': heatmap_ax,
        'distance_matrix': distance_df,
        'heatmap_order': heatmap_order,
        'heatmap_sample_ids': sample_ids_sorted,
        'skbio_distance_matrix': distance_matrix,
        'pcoa': ordination,
    }


def run_permanova(data, groups, metric='jaccard', permutations=999, seed=None):
    """
    Run PERMANOVA on a feature table where columns are samples.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature table where rows are features/taxa and columns are samples.
    groups : pandas.Series, dict, list, tuple, or numpy.ndarray
        Group labels for each sample. If a Series or dict is provided, it is
        aligned to data.columns by sample name. If an array-like object is
        provided, it must have the same order and length as data.columns.
    metric : str, default 'jaccard'
        Distance metric. Supported values are 'jaccard', 'braycurtis', and
        'root_jensen_shannon' (aliases: 'rjsd', 'jensenshannon').
    permutations : int, default 999
        Number of permutations used by PERMANOVA.
    seed : int, numpy.random.Generator, or None
        Random seed passed to scikit-bio's PERMANOVA implementation.

    Returns
    -------
    dict
        Contains the PERMANOVA result, distance matrix dataframe, skbio
        DistanceMatrix, and aligned grouping vector.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with samples in columns")
    if data.shape[1] < 2:
        raise ValueError("data must contain at least two sample columns")

    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    metric_key = metric.lower().replace("-", "_").replace(" ", "_")
    metric_aliases = {
        'jaccard': 'jaccard',
        'braycurtis': 'braycurtis',
        'bray_curtis': 'braycurtis',
        'root_jensen_shannon': 'jensenshannon',
        'root_jensen_shannon_divergence': 'jensenshannon',
        'rjsd': 'jensenshannon',
        'jensenshannon': 'jensenshannon',
    }

    if metric_key not in metric_aliases:
        raise ValueError(
            "metric must be one of: 'jaccard', 'braycurtis', or 'root_jensen_shannon'"
        )

    sample_ids = data.columns.astype(str).tolist()

    if isinstance(groups, pd.Series):
        grouping = groups.reindex(sample_ids)
    elif isinstance(groups, dict):
        grouping = pd.Series(groups).reindex(sample_ids)
    else:
        grouping = pd.Series(groups, index=sample_ids)

    if len(grouping) != len(sample_ids):
        raise ValueError("groups must have the same length as the number of sample columns")
    if grouping.isna().any():
        missing = grouping.index[grouping.isna()].tolist()
        raise ValueError(f"Missing group labels for samples: {missing}")
    if grouping.nunique() < 2:
        raise ValueError("PERMANOVA requires at least two groups")

    scipy_metric = metric_aliases[metric_key]
    sample_matrix = data.to_numpy(dtype=float).T

    if np.any(~np.isfinite(sample_matrix)):
        raise ValueError("data contains NaN or infinite values")
    if scipy_metric in {'braycurtis', 'jensenshannon'} and np.any(sample_matrix < 0):
        raise ValueError(f"{metric} distance requires non-negative values")

    if scipy_metric == 'jaccard':
        distance_input = sample_matrix > 0
    else:
        distance_input = sample_matrix

    distances = squareform(pdist(distance_input, metric=scipy_metric))
    distance_df = pd.DataFrame(distances, index=sample_ids, columns=sample_ids)
    distance_matrix = DistanceMatrix(distance_df.to_numpy(), ids=sample_ids)
    try:
        result = permanova(distance_matrix, grouping.to_numpy(), permutations=permutations, seed=seed)
    except TypeError:
        if seed is not None:
            raise TypeError("The installed scikit-bio version does not support the seed argument for permanova")
        result = permanova(distance_matrix, grouping.to_numpy(), permutations=permutations)

    return {
        'permanova': result,
        'distance_matrix': distance_df,
        'skbio_distance_matrix': distance_matrix,
        'grouping': grouping,
    }


def plot_correlation(x, y, labels, x_title, y_title, x_annotation=0.05, y_annotation=0.95):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)

    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    labels_valid = labels[valid]

    if x_valid.size < 2:
        raise ValueError("At least two finite x/y pairs are required to calculate correlation")

    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    ax_corr.scatter(x_valid, y_valid, s=80, alpha=0.8, color="#3B6FB6")

    pearson_r, pearson_p = pearsonr(x_valid, y_valid)

    for label, xi, yi in zip(labels_valid, x_valid, y_valid):
        ax_corr.text(xi, yi, label, fontsize=8, ha="left", va="bottom")

    fit = np.polyfit(x_valid, y_valid, deg=1)
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    ax_corr.plot(x_line, np.polyval(fit, x_line), color="black", linewidth=1.5)

    ax_corr.set_xlabel(x_title)
    ax_corr.set_ylabel(y_title)
    ax_corr.text(
        x_annotation,
        y_annotation,
        f"Pearson r = {pearson_r:.2f}, p = {pearson_p:.3g}\n",
        transform=ax_corr.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    fig_corr.tight_layout()

    plt.show()
    return {
        "figure": fig_corr,
        "axis": ax_corr,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "valid": valid,
        "fit": fit,
    }

def plot_pcoa_by_groups(data, groups, figsize=(5, 4), ax=None, show_labels=False,
                        legend_loc="best", legend_bbox_to_anchor=None, legend_frame=True, path=None):
    """
    Plot PCoA using Root Jensen-Shannon distance and color samples by group.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature table where rows are features/taxa and columns are samples.
    groups : pandas.Series, dict, list, tuple, or numpy.ndarray
        Group labels for each sample. If a Series or dict is provided, it is
        aligned to data.columns by sample name. If an array-like object is
        provided, it must have the same order and length as data.columns.
    figsize : tuple, default (5, 4)
        Figure size used when ax is not provided.
    ax : matplotlib.axes.Axes or None
        Existing axis to plot on. If None, a new figure and axis are created.
    show_labels : bool, default False
        Whether to draw sample names next to points.
    legend_loc : str, default "best"
        Legend location passed to matplotlib.
    legend_bbox_to_anchor : tuple or None
        Optional bbox_to_anchor passed to matplotlib's legend.
    legend_frame : bool, default True
        Whether to draw a box around the legend.
    path : str, pathlib.Path, or None
        If provided, save the figure to this path.
    """
    sample_ids = data.columns.astype(str).tolist()

    if isinstance(groups, pd.Series):
        grouping = groups[~groups.index.duplicated(keep="first")].reindex(sample_ids)
    elif isinstance(groups, dict):
        grouping = pd.Series(groups).reindex(sample_ids)
    else:
        grouping = pd.Series(groups, index=sample_ids)

    if grouping.isna().any():
        missing = grouping.index[grouping.isna()].tolist()
        raise ValueError(f"Missing group labels for samples: {missing}")

    sample_matrix = data.to_numpy(dtype=float).T
    distances = squareform(pdist(sample_matrix, metric="jensenshannon"))
    distance_df = pd.DataFrame(distances, index=sample_ids, columns=sample_ids)
    distance_matrix = DistanceMatrix(distance_df.to_numpy(), ids=sample_ids)
    ordination = pcoa(distance_matrix)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    coords = ordination.samples.iloc[:, :2]
    var_exp = ordination.proportion_explained.iloc[:2] * 100
    unique_groups = pd.unique(grouping)
    palette = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#000000",  # black
    ]
    colors = [palette[i % len(palette)] for i in range(len(unique_groups))]

    for group, color in zip(unique_groups, colors):
        mask = grouping.to_numpy() == group
        ax.scatter(
            coords.iloc[mask, 0],
            coords.iloc[mask, 1],
            s=80,
            alpha=0.85,
            color=color,
            edgecolor="black",
            label=group,
        )

    if show_labels:
        for sample_id, x, y in zip(sample_ids, coords.iloc[:, 0], coords.iloc[:, 1]):
            ax.text(x, y, sample_id, fontsize=8, ha="left", va="bottom")

    ax.set_xlabel(f"PCoA1 ({var_exp.iloc[0]:.1f}%)")
    ax.set_ylabel(f"PCoA2 ({var_exp.iloc[1]:.1f}%)")
    ax.legend(title="Group", loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, frameon=legend_frame)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")

    return {
        "figure": fig,
        "axis": ax,
        "distance_matrix": distance_df,
        "skbio_distance_matrix": distance_matrix,
        "pcoa": ordination,
        "grouping": grouping,
    }
