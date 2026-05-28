from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "Messaoudene_et_al"
PHYLOGENETIC_RESULTS_DIR = RESULTS_DIR / "phylogenetic_analysis"
FUNCTIONAL_RESULTS_DIR = RESULTS_DIR / "functional_analysis"


def load_phylogenetic_results(base_dir=PHYLOGENETIC_RESULTS_DIR) -> dict:
    base_dir = Path(base_dir)

    with open(base_dir / "phylogenetic_test_results.pkl", "rb") as f:
        phylogenetic_test_results = pickle.load(f)

    return {
        "unifrac_similarities": pd.read_csv(base_dir / "unifrac_similarities.csv", index_col=0),
        "species_categories": pd.read_csv(base_dir / "species_categories.csv"),
        "phylogenetic_test_results": phylogenetic_test_results,
    }


def load_functional_results(base_dir=FUNCTIONAL_RESULTS_DIR) -> dict:
    base_dir = Path(base_dir)

    with open(base_dir / "S_lost_colon.pkl", "rb") as f:
        S_lost_colon = pickle.load(f)
    with open(base_dir / "S_lost_transient_comb.pkl", "rb") as f:
        S_lost_transient_comb = pickle.load(f)
    with open(base_dir / "S_base_colon.pkl", "rb") as f:
        S_base_colon = pickle.load(f)
    with open(base_dir / "S_base_transient_comb.pkl", "rb") as f:
        S_base_transient_comb = pickle.load(f)

    return {
        "AUC_mean_colon_transient_vs_lost_valid": np.load(
            base_dir / "AUC_mean_colon_transient_vs_lost_valid.npy"
        ),
        "adjusted_pvalues_mean_colon_transient_vs_lost_valid": np.load(
            base_dir / "adjusted_pvalues_mean_colon_transient_vs_lost_valid.npy"
        ),
        "AUC_mean_colon_transient_vs_base_valid": np.load(
            base_dir / "AUC_mean_colon_transient_vs_base_valid.npy"
        ),
        "adjusted_pvalues_mean_colon_transient_vs_base_valid": np.load(
            base_dir / "adjusted_pvalues_mean_colon_transient_vs_base_valid.npy"
        ),
        "AUC_mean_colon_transient_vs_lost_valid_subjects": pd.read_csv(
            base_dir / "AUC_mean_colon_transient_vs_lost_valid_subjects.csv",
            index_col=0,
        ).iloc[:, 0],
        "adjusted_pvalues_mean_colon_transient_vs_lost_valid_subjects": pd.read_csv(
            base_dir / "adjusted_pvalues_mean_colon_transient_vs_lost_valid_subjects.csv",
            index_col=0,
        ).iloc[:, 0],
        "AUC_mean_colon_transient_vs_base_valid_subjects": pd.read_csv(
            base_dir / "AUC_mean_colon_transient_vs_base_valid_subjects.csv",
            index_col=0,
        ).iloc[:, 0],
        "adjusted_pvalues_mean_colon_transient_vs_base_valid_subjects": pd.read_csv(
            base_dir / "adjusted_pvalues_mean_colon_transient_vs_base_valid_subjects.csv",
            index_col=0,
        ).iloc[:, 0],
        "valid_subject_results": pd.read_csv(base_dir / "valid_subject_results.csv", index_col=0),
        "S_lost_colon": S_lost_colon,
        "S_lost_transient_comb": S_lost_transient_comb,
        "S_base_colon": S_base_colon,
        "S_base_transient_comb": S_base_transient_comb,
    }


def plot_functional_auc_vs_unifrac(phylogenetic_results: dict, functional_results: dict):
    functional_auc = functional_results["AUC_mean_colon_transient_vs_lost_valid_subjects"]
    eps = 1e-12
    unifrac = np.log2(
        (phylogenetic_results["unifrac_similarities"]["colonizers_lost"] + eps)
        / (phylogenetic_results["unifrac_similarities"]["transient_lost"] + eps)
    )

    aligned = pd.concat(
        [
            functional_auc.rename("Functional AUC"),
            unifrac.rename("UniFrac log2 ratio"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        aligned["Functional AUC"],
        aligned["UniFrac log2 ratio"],
        s=140,
        alpha=0.78,
        color="#3B6FB6",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("Functional AUC: colonizers vs. transient relative to lost species", fontsize=18)
    ax.set_ylabel("log2 UniFrac ratio: colonizers-lost / transient-lost", fontsize=18)
    ax.tick_params(axis="both", labelsize=15, width=1.3, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    fig.tight_layout()
    return fig, ax, aligned


phylogenetic_results = load_phylogenetic_results()
functional_results = load_functional_results()
fig, ax, aligned_results = plot_functional_auc_vs_unifrac(phylogenetic_results, functional_results)
plt.show()

print(0)
