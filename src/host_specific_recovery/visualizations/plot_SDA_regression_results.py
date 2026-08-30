import matplotlib.pyplot as plt


def plot_prediction_scatter(predictions, title, metrics=None, path=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(
        predictions["y_pred"],
        predictions["y_true"],
        s=80,
        alpha=0.85,
        color="#2A9D8F",
        edgecolor="#264653",
        linewidth=0.8,
    )

    if metrics is None:
        pearson = predictions["y_true"].corr(predictions["y_pred"], method="pearson")
        spearman = predictions["y_true"].corr(predictions["y_pred"], method="spearman")
    else:
        pearson = metrics.loc[metrics["metric"] == "pearson_correlation", "value"].iloc[0]
        spearman = metrics.loc[metrics["metric"] == "spearman_correlation", "value"].iloc[0]

    ax.text(
        0.05,
        0.95,
        f"r = {pearson:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )
    ax.set_xlabel("Predicted standardized Jaccard", fontsize=13)
    ax.set_ylabel("Observed standardized Jaccard", fontsize=13)
    ax.tick_params(axis="both", labelsize=12, width=1.2, length=5)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()
