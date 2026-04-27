import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from src.host_specific_recovery.utils.general_utils import benjamini_hochberg, p_to_label, p_to_label_one_sided
from scipy.stats import binomtest

def plot_SDA(outputs, fig_size, sur_color, real_color, show_y=True, legend=True, ymin=None, ymax=None, dir=None,
             y_title='Standardized Jaccard similarity', naive=False):
    """
    Plot SDA results.
    :param outputs: surrogate data analysis outputs dictionary
    :param fig_size: figure size tuple
    :param sur_color: color for surrogate points
    :param real_color: color for observed points
    :param show_y: whether to show y-axis labels
    :param legend: whether to show legend
    :param ymin: minimum y-axis limit
    :param ymax: maximum y-axis limit
    :param dir: directory to save the figure, if None show the plot
    """

    # initialize lists
    obs_norm, surr_norm = [], []

    if naive:
        sim = outputs["similarity_naive"]
        sim_others = outputs["similarity_others_naive"]
    else:
        sim = outputs["similarity_mid"]
        sim_others = outputs["similarity_others_mid"]

    # calculate standardized statistics
    for obs, sur in zip(sim, sim_others):
        sur = np.asarray(sur, dtype=float)
        mu, sd = sur.mean(), sur.std(ddof=0)
        surr_norm.append((sur - mu) / sd)
        obs_norm.append((obs - mu) / sd)

    obs_norm = np.asarray(obs_norm)

    # calculate p-values
    z_scores = obs_norm.copy()
    pvals = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # adjust p-values
    pvals_adj = benjamini_hochberg(pvals)
    pvals_labels = [p_to_label(p, z) for p, z in zip(pvals_adj, z_scores)]

    success_idx = []
    fail_idx = []

    for i, (o, s, l) in enumerate(zip(obs_norm, surr_norm, pvals_labels)):
        if (o > s.max()) and (l != 'ns'):
            success_idx.append(i)
        else:
            fail_idx.append(i)

    # sort each group by descending observed z-score
    success_idx.sort(key=lambda i: -obs_norm[i])
    fail_idx.sort(key=lambda i: -obs_norm[i])
    order = success_idx + fail_idx
    obs_sorted = obs_norm[order]
    surr_sorted = [surr_norm[i] for i in order]
    pvals_adj_sorted = pvals_adj[order]
    pvals_labels_sorted = [pvals_labels[i] for i in order]

    x = np.arange(len(order))
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=fig_size)

    n_success = len(success_idx)
    ax.axvspan(-0.5, n_success - 0.5, color='lightgray', alpha=0.4, zorder=0)

    for xi, sur in zip(x, surr_sorted):
        jitter = rng.uniform(-0.1, 0.1, size=len(sur))
        ax.scatter(np.full_like(sur, xi) + jitter, sur, color=sur_color, s=60, alpha=0.5,
                   label="Surrogate baseline" if xi == 0 else None, zorder=1)

    ax.scatter(x, obs_sorted, color=real_color, s=120, alpha=0.5, label="Baseline", zorder=2)

    if show_y:
        ax.set_ylabel(y_title, fontsize=14)
        plt.yticks(fontsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(3))
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')

    if legend:
        ax.legend(fontsize=10, loc='center right', bbox_to_anchor=(1, 0.7))

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        new_ymax = ymax + 2.25
        ax.set_ylim(ymin, new_ymax)
        label_y = new_ymax - 0.2
    else:
        ymin, ymax = ax.get_ylim()
        new_ymax = ymax + 2.25
        ax.set_ylim(ymin, new_ymax)
        label_y = new_ymax - 0.2

    for xi, l in zip(x, pvals_labels_sorted):
        ax.text(xi, label_y, l, ha='center', va='top', fontsize=10, linespacing=0.6, clip_on=True)

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(x) - 0.5)

    plt.tight_layout()
    if dir is not None:
        plt.savefig(dir, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
    return fig, ax, order, obs_sorted, surr_sorted, pvals_adj_sorted, pvals_labels

def plot_SDA_slow(outputs, significance, fig_size, sur_color, real_color, show_y=True, legend=True, ymin=None,
                  ymax=None, dir=None):
    """
    Plot SDA results.
    :param outputs: surrogate data analysis outputs dictionary
    :param significance: list of significance labels for each observation
    :param fig_size: figure size tuple
    :param sur_color: color for surrogate points
    :param real_color: color for observed points
    :param show_y: whether to show y-axis labels
    :param legend: whether to show legend
    :param ymin: minimum y-axis limit
    :param ymax: maximum y-axis limit
    :param dir: directory to save the figure, if None show the plot
    :return: fig, ax, order, obs_sorted, surr_sorted, pvals_adj_sorted, pvals_labels_sorted
    """

    # initialize lists
    obs_norm, surr_norm = [], []

    sim = outputs["similarity"]
    sim_others = outputs["similarity_others"]

    # calculate standardized statistics
    for obs, sur in zip(sim, sim_others):
        sur = np.asarray(sur, dtype=float)
        mu, sd = sur.mean(), sur.std(ddof=0)
        surr_norm.append((sur - mu) / sd)
        obs_norm.append((obs - mu) / sd)

    obs_norm = np.asarray(obs_norm, dtype=float)

    # calculate p-values
    z_scores = obs_norm.copy()
    pvals = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # adjust p-values
    pvals_adj = benjamini_hochberg(pvals)
    pvals_labels = [p_to_label_one_sided(p) for p in pvals_adj]

    success_idx, near_idx, fail_idx = [], [], []

    for i, (o, s, p) in enumerate(zip(obs_norm, surr_norm, pvals_adj)):
        label = p_to_label_one_sided(p)
        if significance[i] == "ns":
            fail_idx.append(i)
        elif (label != "ns") and (significance[i] != "ns") and (o > np.max(s)):
            success_idx.append(i)
        else:
            near_idx.append(i)

    success_idx.sort(key=lambda i: -obs_norm[i])
    near_idx.sort(key=lambda i: -obs_norm[i])
    fail_idx.sort(key=lambda i: -obs_norm[i])

    order = success_idx + near_idx + fail_idx
    obs_sorted = obs_norm[order]
    surr_sorted = [surr_norm[i] for i in order]
    pvals_adj_sorted = pvals_adj[order]
    pvals_labels_sorted = [pvals_labels[i] for i in order]

    x = np.arange(len(order))
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=fig_size)

    n_success = len(success_idx)
    n_near = len(near_idx)

    if n_success:
        ax.axvspan(-0.5, n_success - 0.5, color="lightpink", alpha=0.4, zorder=0)
    if n_near:
        ax.axvspan(n_success - 0.5, n_success + n_near - 0.5, color="lightgray", alpha=0.4, zorder=0)

    for xi, sur in zip(x, surr_sorted):
        jitter = rng.uniform(-0.1, 0.1, size=len(sur))
        ax.scatter(np.full_like(sur, xi) + jitter, sur, color=sur_color, s=60, alpha=0.5,
            label="Surrogate baseline" if xi == 0 else None, zorder=1)

    ax.scatter(x, obs_sorted, color=real_color, s=120, alpha=0.5, label="Baseline", zorder=2)

    if show_y:
        ax.set_ylabel("Standardized Jaccard similarity", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(3))
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")

    if legend:
        ax.legend(fontsize=10, loc="center right", bbox_to_anchor=(1, 0.7))

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        new_ymax = ymax + 2.25
        ax.set_ylim(ymin, new_ymax)
        label_y = new_ymax - 0.2
    else:
        ymin_auto, ymax_auto = ax.get_ylim()
        new_ymax = ymax_auto + 2.25
        ax.set_ylim(ymin_auto, new_ymax)
        label_y = new_ymax - 0.2

    for xi, l in zip(x, pvals_labels_sorted):
        ax.text(xi, label_y, l,  ha="center", va="top", fontsize=10, linespacing=0.6, clip_on=True)

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(x) - 0.5)

    plt.tight_layout()
    if dir is not None:
        plt.savefig(dir, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()

    return fig, ax, order, obs_sorted, surr_sorted, pvals_adj_sorted, pvals_labels_sorted

def plot_binomial_tests(tests, path=None):
    """
    :param tests : list of dict, each dictionary should contain 'k', 'n', 'p', and 'interval'.
    :param path : str or None. If provided, save the figure to this path. If None, return the figure object.

    Returns:
    matplotlib.figure.Figure or None
        Returns fig if path is None, otherwise saves the figure and returns None.
    """
    n_tests = len(tests)
    x = np.arange(n_tests)

    observed = np.array([t['k'] / t['n'] for t in tests], dtype=float)
    expected = np.array([t['p'] for t in tests], dtype=float)
    lower = np.array([t['interval'][0] / t['n'] for t in tests], dtype=float)
    upper = np.array([t['interval'][1] / t['n'] for t in tests], dtype=float)

    pvals = np.array([
        binomtest(t['k'], t['n'], t['p'], alternative='greater').pvalue
        for t in tests
    ], dtype=float)

    def p_to_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        return ''

    stars = [p_to_stars(p) for p in pvals]

    yerr = np.vstack([expected - lower, upper - expected])

    fig, ax = plt.subplots(figsize=(3, 4))

    ax.errorbar(
        x,
        expected,
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=10,
        markersize=10
    )

    ax.scatter(x, observed, color='blue', s=200)

    for i in range(n_tests):
        if observed[i] > upper[i] or observed[i] < lower[i]:
            ax.scatter(x[i], observed[i], color='#E69F00', s=200, zorder=3)

    y_max = max(np.max(observed), np.max(upper))
    y_star = y_max + 0.05
    ax.set_ylim(-0.05, y_star + 0.075)

    for i in range(n_tests):
        if stars[i]:
            ax.text(x[i], y_star, stars[i], ha='center', va='bottom', fontsize=12)

    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlim(-0.5, n_tests - 0.5)
    ax.set_ylabel('Proportion of successes', fontsize=20)

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return None

    return fig
