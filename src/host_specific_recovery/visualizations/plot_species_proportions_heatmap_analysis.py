import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc


def plot_proportions_heatmap(outputs, weighted=False, path=None):

    if weighted:
        mat = outputs["weighted_proportions_mat"]
    else:
        mat = outputs["proportions_mat"]

    idx = np.argsort(mat[:, 0] + mat[:, 1] + mat[:, 2])

    mat = mat[idx, :]

    n_rows, n_comp = mat.shape

    colors = ['#7f7f7f', '#929292', '#4285FF', '#25478F', '#dc3912']

    fig, ax = plt.subplots(figsize=(6, 8))

    y_pos = np.arange(n_rows)
    left = np.zeros(n_rows)

    titles = [' ', ' ', ' ', ' ', ' ']

    split_x = np.zeros(n_rows)

    for k in range(n_comp):
        patches = ax.barh(
            y_pos, mat[:, k], left=left,
            color=colors[k], edgecolor='white', height=1,
            label=titles[k])
        if k == 2:
            split_x = left + mat[:, k]
        left += mat[:, k]

    ax.vlines(split_x, y_pos - .5, y_pos + .5, color='#F4B400', lw=5, alpha=0)

    bar_h = 1
    half_h = bar_h / 2
    x_eps = 0.015 * (split_x.max() - split_x.min())

    for i in range(n_rows - 1):
        y_h = y_pos[i] + half_h
        if i in [1, 3, 7, 10, 11, 14, 15, 19, 21, 22, 25, 26]:
            ax.hlines(y_h, xmin=split_x[i], xmax=split_x[i+1], color='#F4B400', lw=5, alpha=0)
        else:
            ax.hlines(y_h, xmin=split_x[i], xmax=split_x[i + 1], color='#F4B400', lw=5, alpha=0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.margins(x=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.02)

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()


def plot_pie_chart(outputs, weighted=False, path=None):

    if weighted:
        mean_mat = outputs["weighted_proportions_mat"].mean(axis=0)
    else:
        mean_mat = outputs["proportions_mat"].mean(axis=0)

    titles = [' ', ' ', ' ', ' ', ' ']

    colors = ['#7f7f7f', '#929292', '#4285FF', '#25478F', '#dc3912']

    fig, ax = plt.subplots(figsize=(8, 5))

    wedges, texts, autotexts = ax.pie(mean_mat, labels=titles, autopct='%1.1f%%', startangle=90, colors=colors,
                                      wedgeprops=dict(edgecolor='white', linewidth=2), textprops=dict(fontsize=22))

    left_idx, right_idx = 0, 2

    theta_left = wedges[left_idx].theta1
    theta_right = wedges[right_idx].theta2

    for ang_deg in (theta_left, theta_right):
        ang = np.deg2rad(ang_deg)
        ax.plot([0, np.cos(ang)], [0, np.sin(ang)], lw=8, color='#F4B400', solid_capstyle='butt', alpha=0)

    arc = Arc(xy=(0, 0), width=2, height=2, angle=0, theta1=theta_left, theta2=theta_right, lw=8, color='#F4B400',
              alpha=0)
    ax.add_patch(arc)

    for j, t in enumerate(autotexts):
        if j == 1:
            t.set_fontsize(14)
        else:
            t.set_fontsize(18)
        t.set_color('white')

    ax.axis('equal')
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        plt.show()
