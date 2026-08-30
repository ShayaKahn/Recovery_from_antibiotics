from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.analysis.survived_species_analysis import DenseTimeSeriesSurvivedSpeciesDOA
from src.host_specific_recovery.io.Kurokawa_et_al_loader import load_Kurokawa_et_al_data


MIN_DISSIMILARITY_SPECIES = 3
SURVIVED_SPECIES_MIN_PREVALENCE_C = 1.0
SURVIVED_SPECIES_MIN_PREVALENCE_D = 0.99

# Placeholder until the real baseline sample rows are known.
MOUSE_C_BASELINE_INDEXES = range(0, 89)
MOUSE_D_BASELINE_INDEXES = range(0, 89)


def run_Kurokawa_survived_species_doa(
        mouse_c_baseline_indexes=MOUSE_C_BASELINE_INDEXES,
        mouse_d_baseline_indexes=MOUSE_D_BASELINE_INDEXES,
        min_dissimilarity_species=MIN_DISSIMILARITY_SPECIES,
        similarity_method="jaccard",
        dissimilarity_method="rjsd",
        survived_species_min_prevalence_c=SURVIVED_SPECIES_MIN_PREVALENCE_C,
        survived_species_min_prevalence_d=SURVIVED_SPECIES_MIN_PREVALENCE_D
):
    dataset = load_Kurokawa_et_al_data(normalize=True)

    mouse_c_analysis = DenseTimeSeriesSurvivedSpeciesDOA(
        samples=dataset["mouseC_relative_abundance"],
        baseline_indexes=mouse_c_baseline_indexes,
        min_dissimilarity_species=min_dissimilarity_species,
        similarity_method=similarity_method,
        dissimilarity_method=dissimilarity_method,
        survived_species_min_prevalence=survived_species_min_prevalence_c,
    )
    mouse_c_results = mouse_c_analysis.calculate()

    mouse_d_analysis = DenseTimeSeriesSurvivedSpeciesDOA(
        samples=dataset["mouseD_relative_abundance"],
        baseline_indexes=mouse_d_baseline_indexes,
        min_dissimilarity_species=min_dissimilarity_species,
        similarity_method=similarity_method,
        dissimilarity_method=dissimilarity_method,
        survived_species_min_prevalence=survived_species_min_prevalence_d,
    )
    mouse_d_results = mouse_d_analysis.calculate()

    return {
        "mouseC_baseline_indexes": np.asarray(list(mouse_c_baseline_indexes), dtype=int),
        "mouseD_baseline_indexes": np.asarray(list(mouse_d_baseline_indexes), dtype=int),
        "mouseC": mouse_c_results,
        "mouseD": mouse_d_results,
    }


if __name__ == "__main__":
    results = run_Kurokawa_survived_species_doa(similarity_method="overlap", dissimilarity_method="rjsd",)

    for mouse_id in ("mouseC", "mouseD"):
        mouse_results = results[mouse_id]
        print(
            mouse_id,
            "similarity_values",
            mouse_results["similarity_values"].shape,
            "dissimilarity_values",
            mouse_results["dissimilarity_values"].shape,
            "n_survived_species",
            mouse_results["n_survived_species"],
        )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, mouse_id, color in zip(axes, ["mouseC", "mouseD"], ["#2A9D8F", "#E76F51"]):
        mouse_results = results[mouse_id]
        ax.scatter(
            mouse_results["similarity_values"],
            mouse_results["dissimilarity_values"],
            s=28,
            alpha=0.75,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=mouse_id,
        )

        ax.set_title(mouse_id)
        ax.set_xlabel("Similarity to baseline")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Dissimilarity to baseline")
    fig.tight_layout()
    plt.show()

    print(0)