import numpy as np
import matplotlib.pyplot as plt


def plot_colonization_probabilities_bubble(outputs, color, path=None, size_scale=50, alpha=0.7):

    probs = outputs["probs"]

    vals = np.array(list(probs.values()), dtype=float)
    x = vals[:, 0]
    y = vals[:, 1]

    pairs = np.column_stack([x, y])
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)

    x_u = uniq[:, 0]
    y_u = uniq[:, 1]

    s = size_scale * counts

    plt.figure(figsize=(8, 8))
    plt.scatter(y_u, x_u, s=s, alpha=alpha, c=color)

    # Axis labels
    plt.xlabel("Colonization probability | New", fontsize=18, labelpad=18)
    plt.ylabel("Colonization probability | Returned", fontsize=18, labelpad=18)

    # Set limits *before* drawing the diagonal
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    # Diagonal line y = x
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    lo = max(xmin, ymin)
    hi = min(xmax, ymax)
    plt.plot([lo, hi], [lo, hi],
             linestyle="--",
             linewidth=2.8,
             alpha=0.8,
             color="black")

    plt.tick_params(axis="both", which="major",
                    length=10, width=2.5,
                    direction="out", labelsize=20)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)
    else:
        plt.show()

def plot_cross_species(outputs, title, path=None):

    new_probs = outputs["new_probs"]
    returned_probs = outputs["returned_probs"]

    eps = 0.05

    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    marker_area = 250
    line_width = 2.8
    axis_width = 3

    ax.scatter(new_probs, returned_probs,
               s=marker_area, marker='o', c='#1A71B8', alpha=0.7, linewidths=0, label=title)

    xline = np.array([-eps, 1 + eps])
    ax.plot(xline, xline, '--', color='black', linewidth=line_width)

    ax.set_xlim(-eps, 1 + eps)
    ax.set_ylim(-eps, 1 + eps)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=20)

    ax.set_xlabel("Colonization probability of the New species", fontsize=18, labelpad=18)
    ax.set_ylabel("Recolonization probability of the Returned species", fontsize=18, labelpad=18)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
        spine.set_color('black')

    ax.tick_params(axis='both', which='major', length=10, width=2.5, direction='out')

    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300)
    else:
        plt.show()
