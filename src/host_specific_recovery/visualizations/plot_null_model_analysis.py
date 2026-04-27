import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from src.host_specific_recovery.utils.general_utils import benjamini_hochberg, p_to_label


def plot_NM_violin(outputs, fig_size, sur_color, real_color,
                   custom_order=None, show_y=True, legend=True, ymin=None, ymax=None, dir=None):
    """
    Plot NM specificity results with standardized specificity scores.
    :param outputs: outputs from null model analysis
    :param fig_size: figure size tuple
    :param sur_color: color for surrogate points
    :param real_color: color for observed points
    :param custom_order: custom order for subjects
    :param show_y: whether to show y-axis labels
    :param legend: whether to show legend
    :param ymin: minimum y-axis limit
    :param ymax: maximum y-axis limit
    :param dir: directory to save the figure, if None show the plot:
    :return: fig, ax
    """

    # initialize lists
    obs_norm, surr_norm = [], []

    real_dist = outputs['real_dist']
    shuffled_dist = outputs['shuffled_dist']

    # calculate standardized statistics
    for obs, sur in zip(real_dist, shuffled_dist):
        sur = np.asarray(sur, dtype=float)
        mu, sd = sur.mean(), sur.std(ddof=0)
        surr_norm.append((sur - mu) / sd)
        obs_norm.append((obs - mu) / sd)

    obs_norm = np.asarray(obs_norm)

    # calculate p-values
    z_scores = obs_norm.copy()
    pvals = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # adjust p-values using Benjamini-Hochberg
    pvals_adj = benjamini_hochberg(pvals)

    if custom_order is not None:
        order = custom_order
    else:
        order = np.argsort(-obs_norm)

    obs_sorted = obs_norm[order]
    surr_sorted = [surr_norm[i] for i in order]
    pvals_adj_sorted = pvals_adj[order]

    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=fig_size)

    n_violins = len(surr_sorted)
    quantiles = [[0.25, 0.5, 0.75]] * n_violins

    vp = ax.violinplot(surr_sorted, positions=x, widths=0.8, showmeans=False, showmedians=True, showextrema=True,
                       quantiles=quantiles)

    for body in vp['bodies']:
        body.set_facecolor(sur_color)
        body.set_edgecolor(sur_color)
        body.set_linewidth(1.0)
        body.set_alpha(0.8)

    for name in ('cmedians', 'cmins', 'cmaxes', 'cbars', 'cquantiles'):
        artist = vp.get(name)
        if artist is not None:
            artist.set_color("black")
            artist.set_linewidth(1.0)
            if name == 'cquantiles':
                artist.set_linewidth(0.8)

    ax.scatter(x, obs_sorted, color=real_color, s=120, alpha=0.8, zorder=2)

    if show_y:
        ax.set_ylabel('Standardized Specificity', fontsize=14)
        plt.yticks(fontsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(4))
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xlabel('Subjects', fontsize=16)
    ax.grid(False)

    legend_elements = [
        Patch(facecolor=sur_color, edgecolor=sur_color, label='Synthetic post-ABX samples', alpha=0.8),

    Line2D([0], [0], marker='o', color=real_color, linestyle='None', alpha=0.8, markersize=np.sqrt(120),
           label='Real post-ABX sample')]

    if legend:
        ax.legend(handles=legend_elements, fontsize=10, loc='center right', bbox_to_anchor=(1, 0.7))

    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
        new_ymax = ymax + 2.25
        ax.set_ylim(ymin, new_ymax)
        label_y = new_ymax - 0.2
    else:
        ymin, ymax = ax.get_ylim()
        print(ymin, ymax)
        new_ymax = ymax + 2.25
        ax.set_ylim(ymin, new_ymax)
        label_y = new_ymax - 0.2

    for xi, p, z in zip(x, pvals_adj_sorted, obs_sorted):
        label = p_to_label(p, z)
        ax.text(xi, label_y, label, ha='center', va='top', fontsize=10, linespacing=0.6, clip_on=True)

    plt.tight_layout()
    if dir is not None:
        plt.savefig(dir, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
    return fig, ax
