from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.analysis.survived_species_analysis import run_survived_species_doa_analysis
from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data


MIN_DISSIMILARITY_SPECIES = 10


def plot_similarity_and_rjsd(similarity_values, rjsd_values, x_labels, similarity_label, path=None):
    similarity_values = np.asarray(similarity_values, dtype=float)
    rjsd_values = np.asarray(rjsd_values, dtype=float)
    valid_rjsd_mask = np.isfinite(rjsd_values).all(axis=1)
    similarity_values_valid = similarity_values[valid_rjsd_mask]
    rjsd_values_valid = rjsd_values[valid_rjsd_mask]
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    panel_specs = [
        (axes[0], similarity_values_valid, similarity_label, "#2A9D8F"),
        (axes[1], rjsd_values_valid, "rJSD to baseline", "#E76F51"),
    ]

    for ax, values, ylabel, color in panel_specs:
        for subject_values in values:
            ax.plot(x, subject_values, color=color, alpha=0.25, linewidth=1)
            ax.scatter(x, subject_values, color=color, alpha=0.45, s=22)

        mean_values = values.mean(axis=0)
        ax.plot(x, mean_values, color="black", linewidth=2.5)
        ax.scatter(x, mean_values, color="white", edgecolor="black", linewidth=1.2, s=55, zorder=3)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.tick_params(axis="both", labelsize=11, width=1.1, length=4)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()


def plot_similarity_vs_dissimilarity(
        similarity_values,
        dissimilarity_values,
        x_labels,
        similarity_label,
        dissimilarity_label,
        path=None,
):
    similarity_values = np.asarray(similarity_values, dtype=float)
    dissimilarity_values = np.asarray(dissimilarity_values, dtype=float)
    valid_subject_mask = (
        np.isfinite(similarity_values).all(axis=1)
        & np.isfinite(dissimilarity_values).all(axis=1)
    )
    similarity_values = similarity_values[valid_subject_mask]
    dissimilarity_values = dissimilarity_values[valid_subject_mask]
    if len(similarity_values) == 0:
        raise ValueError("No subjects satisfy the min-species/finite-value filter.")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(x_labels)))

    for subject_idx, (subject_similarity, subject_dissimilarity) in enumerate(
            zip(similarity_values, dissimilarity_values)
    ):
        ax.plot(
            subject_similarity,
            subject_dissimilarity,
            color="0.75",
            alpha=0.35,
            linewidth=1,
            zorder=1,
        )

        for time_idx, label in enumerate(x_labels):
            ax.scatter(
                subject_similarity[time_idx],
                subject_dissimilarity[time_idx],
                color=colors[time_idx],
                edgecolor="white",
                linewidth=0.5,
                s=38,
                alpha=0.9,
                label=label if subject_idx == 0 else None,
                zorder=2,
            )

    for time_idx, label in enumerate(x_labels):
        ax.scatter(
            np.nanmean(similarity_values[:, time_idx]),
            np.nanmean(dissimilarity_values[:, time_idx]),
            color=colors[time_idx],
            edgecolor="black",
            linewidth=1.1,
            s=95,
            marker="D",
            zorder=2,
        )

    ax.set_xlabel(similarity_label, fontsize=13)
    ax.set_ylabel(dissimilarity_label, fontsize=13)
    ax.tick_params(axis="both", labelsize=11, width=1.1, length=4)
    ax.legend(title="Timepoint", frameon=False, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()


def plot_subject_subplots_similarity_vs_dissimilarity(
        similarity_values,
        dissimilarity_values,
        subject_ids,
        x_labels,
        similarity_label,
        dissimilarity_label,
        n_cols=4,
        path=None,
):
    similarity_values = np.asarray(similarity_values, dtype=float)
    dissimilarity_values = np.asarray(dissimilarity_values, dtype=float)
    subject_ids = np.asarray(subject_ids)
    valid_point_mask = np.isfinite(similarity_values) & np.isfinite(dissimilarity_values)
    valid_subject_mask = valid_point_mask.any(axis=1)
    similarity_values = similarity_values[valid_subject_mask]
    dissimilarity_values = dissimilarity_values[valid_subject_mask]
    valid_point_mask = valid_point_mask[valid_subject_mask]
    subject_ids = subject_ids[valid_subject_mask]
    if len(subject_ids) == 0:
        raise ValueError("No subjects satisfy the min-species/finite-value filter.")

    n_subjects = len(subject_ids)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, subject_id, subject_similarity, subject_dissimilarity, subject_valid_points in zip(
            axes,
            subject_ids,
            similarity_values,
            dissimilarity_values,
            valid_point_mask,
    ):
        subject_similarity = subject_similarity[subject_valid_points]
        subject_dissimilarity = subject_dissimilarity[subject_valid_points]
        ax.scatter(
            subject_similarity,
            subject_dissimilarity,
            color="#2A9D8F",
            edgecolor="white",
            linewidth=0.5,
            s=35,
            alpha=0.95,
        )

        ax.set_title(str(subject_id), fontsize=9)
        x_min, x_max = np.nanmin(subject_similarity), np.nanmax(subject_similarity)
        y_min, y_max = np.nanmin(subject_dissimilarity), np.nanmax(subject_dissimilarity)
        x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.05
        y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)

    for ax in axes[n_subjects:]:
        ax.axis("off")

    fig.supxlabel(similarity_label, fontsize=13)
    fig.supylabel(dissimilarity_label, fontsize=13)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()


dataset = load_Messaoudene_et_al_data()

subject_ids = dataset["filtered_keys"]
times_post_abx = dataset["times_post_abx"]

jaccard_rjsd_results = run_survived_species_doa_analysis(
    dataset,
    min_dissimilarity_species=MIN_DISSIMILARITY_SPECIES,
    similarity_method="jaccard",
    dissimilarity_method="rjsd",
)
overlap_rjsd_results = run_survived_species_doa_analysis(
    dataset,
    min_dissimilarity_species=MIN_DISSIMILARITY_SPECIES,
    similarity_method="overlap",
    dissimilarity_method="rjsd",
)
jaccard_values = jaccard_rjsd_results["similarity_values"]
rjsd_values = jaccard_rjsd_results["dissimilarity_values"]
overlap_values = overlap_rjsd_results["similarity_values"]

plot_dir = PROJECT_ROOT / "results" / "Messaoudene_et_al" / "dissimilarity_overlap_analysis"
plot_dir.mkdir(parents=True, exist_ok=True)
x_labels = ["ABX"] + [f"Day {time}" for time in times_post_abx]
plot_similarity_and_rjsd(
    jaccard_values,
    rjsd_values,
    x_labels,
    similarity_label="Jaccard to baseline",
    path=None#plot_dir / "survived_species_jaccard_rjsd.png",
)
plot_subject_subplots_similarity_vs_dissimilarity(
    jaccard_values,
    rjsd_values,
    subject_ids,
    x_labels,
    similarity_label="Jaccard similarity to baseline",
    dissimilarity_label="rJSD to baseline",
    path=None#plot_dir / "survived_species_jaccard_rjsd_scatter.png",
)
plot_subject_subplots_similarity_vs_dissimilarity(
    overlap_values,
    rjsd_values,
    subject_ids,
    x_labels,
    similarity_label="Overlap with baseline",
    dissimilarity_label="rJSD to baseline",
    path=None#plot_dir / "survived_species_overlap_rjsd_scatter.png",
)

print(0)
