from pathlib import Path
import matplotlib.pyplot as plt


AXIS_LABELS = {
    "colonizers_lost": "Colonizers vs. Lost species",
    "transient_lost": "Transient species vs. Lost species",
    "colonizers_other_baseline": "Colonizers vs. Other baseline species",
    "transient_other_baseline": "Transient species vs. Other baseline species",
}


def _plot_unifrac_scatter(similarities, x_col, y_col, path=None):
    data = similarities[[x_col, y_col]].dropna()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data[x_col], data[y_col], s=140, alpha=0.78, color="#3B6FB6", edgecolor="black", linewidth=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.6)
    ax.set_xlabel(AXIS_LABELS.get(x_col, x_col), fontsize=18)
    ax.set_ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=15, width=1.3, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)

    return fig, ax


def plot_unifrac_similarity_scatter(outputs, path=None):
    similarities = outputs["unifrac_similarities"]
    path = Path(path) if path is not None else None

    colonizers_lost_fig, colonizers_lost_ax = _plot_unifrac_scatter(
        similarities,
        "colonizers_lost",
        "transient_lost",
        None if path is None else path / "colonizers_lost_vs_transient_lost.png",
    )
    baseline_fig, baseline_ax = _plot_unifrac_scatter(
        similarities,
        "colonizers_other_baseline",
        "transient_other_baseline",
        None if path is None else path / "colonizers_other_baseline_vs_transient_other_baseline.png",
    )

    return {
        "colonizers_lost_vs_transient_lost": (colonizers_lost_fig, colonizers_lost_ax),
        "colonizers_other_baseline_vs_transient_other_baseline": (baseline_fig, baseline_ax),
    }
