import numpy as np
import matplotlib.pyplot as plt


def connect_columns_plot(outputs, *, labels=None, s=25, mean_s=45, alpha_points=0.35,
                         alpha_lines=0.25, mean_alpha=0.95, lw=1.0, mean_lw=2.0,
                         figsize=(6, 4), ax=None, nan_policy="omit", tight_layout=True,
                         bottom_pad=0.18, tick_labelsize=12, tick_size=6, tick_width=1.2,
                         x_title="Days after ABX", y_title="Number of new species per day", path=None):

    X = np.asarray(outputs["new_species_counts_all_rate"], dtype=float)
    x = np.asarray(outputs["times"], dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_x, n_cols = X.shape
    if x.size != n_x:
        raise ValueError(f"x must have length {n_x}, got {x.size}")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for j in range(n_cols):
        y = X[:, j]
        m = np.isfinite(x) & np.isfinite(y)

        if nan_policy == "omit":
            if m.sum() >= 2:
                ax.plot(x[m], y[m], lw=lw, alpha=alpha_lines, color="0.6")
            if m.sum() >= 1:
                ax.scatter(x[m], y[m], s=s, alpha=alpha_points, color="0.6")

        elif nan_policy == "break":
            ax.scatter(x[m], y[m], s=s, alpha=alpha_points, color="0.6")
            idx = np.where(m)[0]
            if idx.size >= 2:
                splits = np.where(np.diff(idx) != 1)[0] + 1
                for seg in np.split(idx, splits):
                    if seg.size >= 2:
                        ax.plot(x[seg], y[seg], lw=lw, alpha=alpha_lines, color="0.6")
        else:
            raise ValueError("nan_policy must be 'omit' or 'break'")

    mean_y = np.nanmean(X, axis=1)
    m_mean = np.isfinite(x) & np.isfinite(mean_y)
    if m_mean.sum() >= 2:
        ax.plot(x[m_mean], mean_y[m_mean], lw=mean_lw, alpha=mean_alpha, color="black", zorder=3)
    if m_mean.sum() >= 1:
        ax.scatter(x[m_mean], mean_y[m_mean], s=mean_s, alpha=mean_alpha, color="black", zorder=4)

    ax.set_xlabel(x_title, fontsize=14, labelpad=10)
    ax.set_ylabel(y_title, fontsize=14, labelpad=8)

    ticks = np.unique(x[np.isfinite(x)])
    ticks.sort()
    ax.set_xticks(ticks)

    if labels is not None:
        if len(labels) != n_x:
            raise ValueError("labels length must match len(x)")
        if np.unique(x).size != x.size:
            raise ValueError("labels requires x to have unique values (same length, no repeats)")
        ax.set_xticklabels(labels)

    ax.tick_params(axis="both", which="major",
                   labelsize=tick_labelsize, length=tick_size, width=tick_width)

    if fig is not None:
        fig.subplots_adjust(bottom=bottom_pad)
        if tight_layout:
            fig.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')

    return ax
