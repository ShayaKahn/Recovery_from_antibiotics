from pathlib import Path
import numpy as np
import csv
import os
import pickle
import pandas as pd


def save_sda_results(outputs: dict, base_dir: str | Path) -> None:

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    similarities = []
    for d in outputs["results"]:
        sims = list(d.values())
        for s in sims:
            similarities.append(s)

    np.save(base_dir / "similarities.npy", similarities)

    similarities_mid = []
    for d in outputs["results_mid"]:
        sims = list(d.values())
        for s in sims:
            similarities_mid.append(s)

    np.save(base_dir / "similarities_mid.npy", similarities_mid)

    similarities_naive = []
    for d in outputs["results_naive"]:
        sims = list(d.values())
        for s in sims:
            similarities_naive.append(s)

    np.save(base_dir / "similarities_naive.npy", similarities_naive)
    np.save(base_dir / "similarity.npy", np.array(outputs["similarity"]))
    np.save(base_dir / "similarity_others.npy", np.array(outputs["similarity_others"]))
    np.save(base_dir / "similarity_mid.npy", np.array(outputs["similarity_mid"]))
    np.save(base_dir / "similarity_others_mid.npy", np.array(outputs["similarity_others_mid"]))
    np.save(base_dir / "similarity_naive.npy", np.array(outputs["similarity_naive"]))
    np.save(base_dir / "similarity_others_naive.npy", np.array(outputs["similarity_others_naive"]))
    np.save(base_dir / "ranks.npy", np.array(outputs["ranks"]))
    np.save(base_dir / "ranks_mid.npy", np.array(outputs["ranks_mid"]))
    np.save(base_dir / "ranks_naive.npy", np.array(outputs["ranks_naive"]))


def save_sda_plot_results(results_sda_plot: dict, base_dir: str | Path, slow=False) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    figure = results_sda_plot.get("figure")
    if figure is not None:
        if slow:
            figure.savefig(base_dir / "sda_plot_slow.png", dpi=300, bbox_inches="tight", facecolor="white")
        else:
            figure.savefig(base_dir / "sda_plot.png", dpi=300, bbox_inches="tight", facecolor="white")
    if slow:
        np.save(base_dir / "order_slow.npy", np.asarray(results_sda_plot["order"]))
        np.save(base_dir / "obs_sorted_slow.npy", np.asarray(results_sda_plot["obs_sorted"], dtype=float))
        np.save(base_dir / "surr_sorted_slow.npy", np.asarray(results_sda_plot["surr_sorted"], dtype=object))
        np.save(base_dir / "pvals_adj_sorted_slow.npy", np.asarray(results_sda_plot["pvals_adj_sorted"], dtype=float))
        np.save(base_dir / "pvals_labels_sorted_slow.npy", np.asarray(results_sda_plot["pvals_labels_sorted"],
                                                                      dtype=str))
    else:
        np.save(base_dir / "order.npy", np.asarray(results_sda_plot["order"]))
        np.save(base_dir / "obs_sorted.npy", np.asarray(results_sda_plot["obs_sorted"], dtype=float))
        np.save(base_dir / "surr_sorted.npy", np.asarray(results_sda_plot["surr_sorted"], dtype=object))
        np.save(base_dir / "pvals_adj_sorted.npy", np.asarray(results_sda_plot["pvals_adj_sorted"], dtype=float))
        np.save(base_dir / "pvals_labels_sorted.npy", np.asarray(results_sda_plot["pvals_labels_sorted"], dtype=str))

def save_binomial_test_results(outputs: dict, base_dir: str | Path) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    confidence_interval = outputs["confidence_interval"]
    confidence_interval_values = np.array(
        [confidence_interval.low, confidence_interval.high],
        dtype=float
    )

    np.save(base_dir / "binomial_test_p_value.npy", np.array(outputs["p_value"], dtype=float))
    np.save(base_dir / "binomial_test_confidence_interval.npy", confidence_interval_values)
    np.save(base_dir / "binomial_test_interval_null.npy", np.asarray(outputs["interval_null"], dtype=float))

    with open(base_dir / "binomial_test_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p_value", "confidence_interval_low", "confidence_interval_high",
                         "interval_null_low", "interval_null_high"])
        writer.writerow([
            outputs["p_value"],
            confidence_interval.low,
            confidence_interval.high,
            outputs["interval_null"][0],
            outputs["interval_null"][1]
        ])

def save_nm_results(outputs: dict, base_dir: str | Path) -> None:

    with open(base_dir / "real_sim_container.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(outputs["filtered_keys"])
        writer.writerow(outputs["real_sim_container"])

    num_rows = len(outputs["shuffled_sim_container"][0])

    with open(base_dir / "shuffled_sim_container.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(outputs["filtered_keys"])

        for i in range(num_rows):
            row = []
            for k in range(len(outputs["filtered_keys"])):
                row.append(outputs["shuffled_sim_container"][k][i])
            writer.writerow(row)


def save_sc_results(outputs: dict, base_dir: str | Path) -> None:

    # Save results to CSV

    filename_new = "similarity_matrix_new.csv"
    filename_others = "similarity_matrix_others.csv"
    filename_sizes = "sizes_matrix.csv"

    full_path_new = os.path.join(base_dir, filename_new)
    full_path_others = os.path.join(base_dir, filename_others)
    full_path_sizes = os.path.join(base_dir, filename_sizes)

    np.savetxt(full_path_new, outputs["similarity_new"], delimiter=",", fmt="%s")
    np.savetxt(full_path_others, outputs["similarity_others"], delimiter=",", fmt="%s")
    np.savetxt(full_path_sizes, outputs["sizes"], delimiter=",", fmt="%s")

def save_HS_simulation_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "post_sim.npy", outputs["post_sim"])
    np.save(base_dir / "post_sim_others.npy", outputs["post_sim_others"])
    np.save(base_dir / "abx_sim.npy", outputs["abx_sim"])

    np.save(base_dir / "post_sim_off.npy", outputs["post_sim_off"])
    np.save(base_dir / "post_sim_others_off.npy", outputs["post_sim_others_off"])
    np.save(base_dir / "abx_sim_off.npy", outputs["abx_sim_off"])

def save_assembly_times_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "percentage_returned_lst.npy", outputs["percentage_returned_lst"])
    np.save(base_dir / "percentage_new_lst.npy", outputs["percentage_new_lst"])
    np.save(base_dir / "new_counts.npy", outputs["new_counts"])
    np.save(base_dir / "returned_counts.npy", outputs["returned_counts"])

def save_cross_species_probability_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "new_probs.npy", outputs["new_probs"])
    np.save(base_dir / "returned_probs.npy", outputs["returned_probs"])

def save_subject_species_probability_results(outputs: dict, base_dir: str | Path) -> None:

    probs = outputs["probs"]
    abundances = outputs["abundances"]

    file_path = base_dir / "subject_species_probability.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump({'probs': probs, 'abundances': abundances}, f)


def save_functional_data_pipeline_prepare_results(outputs: dict, base_dir: str | Path) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    with open(base_dir / "M_base_no_lost_taxa_lst_PATH.pkl", 'wb') as f:
        pickle.dump(outputs["M_base_no_lost_taxa_lst_PATH"], f)

    with open(base_dir / "M_lost_taxa_lst_PATH.pkl", 'wb') as f:
        pickle.dump(outputs["M_lost_taxa_lst_PATH"], f)

    with open(base_dir / "M_new_taxa_lst_PATH.pkl", 'wb') as f:
        pickle.dump(outputs["M_new_taxa_lst_PATH"], f)

    with open(base_dir / "M_new_taxa_lst_PATH_dict.pkl", 'wb') as f:
        pickle.dump(outputs["M_new_taxa_lst_PATH_dict"], f)

    if "fun_redundancy_lst" in outputs:
        fun_redundancy = np.asarray(outputs["fun_redundancy_lst"], dtype=float)
        if fun_redundancy.ndim == 1:
            fun_redundancy = fun_redundancy[:, np.newaxis]
        sample_labels = [f"sample_{i}" for i in range(fun_redundancy.shape[1])]
        np.save(base_dir / "fun_redundancy_lst.npy", fun_redundancy)
        pd.DataFrame(fun_redundancy, columns=sample_labels).to_csv(
            base_dir / "fun_redundancy_lst.csv",
            index=False,
        )

    if "specificity_similarity_lst" in outputs:
        specificity_similarity = np.asarray(outputs["specificity_similarity_lst"], dtype=float)
        if specificity_similarity.ndim == 1:
            specificity_similarity = specificity_similarity[:, np.newaxis]
        sample_labels = [f"sample_{i}" for i in range(specificity_similarity.shape[1])]
        np.save(base_dir / "specificity_similarity_lst.npy", specificity_similarity)
        pd.DataFrame(specificity_similarity, columns=sample_labels).to_csv(
            base_dir / "specificity_similarity_lst.csv",
            index=False,
        )


def save_functional_data_pipeline_analysis_results(outputs: dict, base_dir: str | Path) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    np.save(base_dir / "AUC_mean_colon_transient_vs_lost_valid.npy", outputs["AUC_mean_colon_transient_vs_lost_valid"])
    np.save(base_dir / "adjusted_pvalues_mean_colon_transient_vs_lost_valid.npy",
            outputs["adjusted_pvalues_mean_colon_transient_vs_lost_valid"])
    np.save(base_dir / "AUC_mean_colon_transient_vs_base_valid.npy", outputs["AUC_mean_colon_transient_vs_base_valid"])
    np.save(base_dir / "adjusted_pvalues_mean_colon_transient_vs_base_valid.npy",
            outputs["adjusted_pvalues_mean_colon_transient_vs_base_valid"])
    np.save(base_dir / "fun_redundancy_valid.npy", outputs["fun_redundancy_valid"])
    np.save(base_dir / "mean_S_lost_base_valid.npy", outputs["mean_S_lost_base_valid"])

    outputs["AUC_mean_colon_transient_vs_lost_valid_subjects"].to_csv(
        base_dir / "AUC_mean_colon_transient_vs_lost_valid_subjects.csv",
        header=True,
    )
    outputs["adjusted_pvalues_mean_colon_transient_vs_lost_valid_subjects"].to_csv(
        base_dir / "adjusted_pvalues_mean_colon_transient_vs_lost_valid_subjects.csv",
        header=True,
    )
    outputs["AUC_mean_colon_transient_vs_base_valid_subjects"].to_csv(
        base_dir / "AUC_mean_colon_transient_vs_base_valid_subjects.csv",
        header=True,
    )
    outputs["adjusted_pvalues_mean_colon_transient_vs_base_valid_subjects"].to_csv(
        base_dir / "adjusted_pvalues_mean_colon_transient_vs_base_valid_subjects.csv",
        header=True,
    )
    outputs["fun_redundancy_valid_subjects"].to_csv(
        base_dir / "fun_redundancy_valid_subjects.csv",
        header=True,
    )
    outputs["mean_S_lost_base_valid_subjects"].to_csv(
        base_dir / "mean_S_lost_base_valid_subjects.csv",
        header=True,
    )
    outputs["valid_subject_results"].to_csv(base_dir / "valid_subject_results.csv")

    with open(base_dir / "S_lost_colon.pkl", 'wb') as f:
        pickle.dump(outputs["S_lost_colon"], f)

    with open(base_dir / "S_lost_transient_comb.pkl", 'wb') as f:
        pickle.dump(outputs["S_lost_transient_comb"], f)

    with open(base_dir / "S_lost_base.pkl", 'wb') as f:
        pickle.dump(outputs["S_lost_base"], f)

    with open(base_dir / "S_base_colon.pkl", 'wb') as f:
        pickle.dump(outputs["S_base_colon"], f)

    with open(base_dir / "S_base_transient_comb.pkl", 'wb') as f:
        pickle.dump(outputs["S_base_transient_comb"], f)


def save_phylogenetic_test_results(outputs: dict, base_dir: str | Path) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    outputs["unifrac_similarities"].to_csv(base_dir / "unifrac_similarities.csv")

    category_rows = []
    for subject, subject_outputs in outputs["results"].items():
        for category, taxa in subject_outputs["categories"].items():
            for taxon in sorted(taxa):
                category_rows.append({
                    "subject": subject,
                    "category": category,
                    "taxon": taxon,
                })

    pd.DataFrame(category_rows, columns=["subject", "category", "taxon"]).to_csv(
        base_dir / "species_categories.csv",
        index=False,
    )

    serializable_results = {
        subject: {
            "categories": subject_outputs["categories"],
            "unifrac_similarities": subject_outputs["unifrac_similarities"],
        }
        for subject, subject_outputs in outputs["results"].items()
    }
    with open(base_dir / "phylogenetic_test_results.pkl", "wb") as f:
        pickle.dump(serializable_results, f)


def save_species_proportions_heatmap_analysis_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "weighted_proportions.npy", outputs["weighted_proportions"])
    np.save(base_dir / "proportions.npy", outputs["proportions"])

def save_survived_species_analysis_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "results_matrix.npy", outputs["results_matrix"])
    np.save(base_dir / "mean.npy", outputs["mean"])


def write_hc(outputs: dict, base_dir: str | Path, model_name: str = "glv") -> None:

    sizes = outputs["sizes"]
    sims_new = outputs["sims_new"]
    sims_survived = outputs["sims_survived"]

    # Save to CSV
    folder = base_dir#"C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/"
    os.makedirs(folder, exist_ok=True)
    filename = f"sizes_{model_name}_test_.csv"
    filename_new = f"sims_new_{model_name}_test_.csv"
    filename_survived = f"sims_survived_{model_name}_test_.csv"

    full_path = os.path.join(folder, filename)
    full_path_new = os.path.join(folder, filename_new)
    full_path_survived = os.path.join(folder, filename_survived)

    np.savetxt(full_path, sizes, delimiter=",", fmt="%s")
    np.savetxt(full_path_new, sims_new, delimiter=",", fmt="%s")
    np.savetxt(full_path_survived, sims_survived, delimiter=",", fmt="%s")

    if model_name != "cr":
        sizes_off = outputs["sizes_off"]
        sims_new_off = outputs["sims_new_off"]
        sims_survived_off = outputs["sims_survived_off"]

        filename_off = f"sizes_{model_name}_off_test_.csv"
        filename_new_off = f"sims_new_{model_name}_off_test_.csv"
        filename_survived_off = f"sims_survived_{model_name}_off_test_.csv"

        full_path_off = os.path.join(folder, filename_off)
        full_path_new_off = os.path.join(folder, filename_new_off)
        full_path_survived_off = os.path.join(folder, filename_survived_off)

        np.savetxt(full_path_off, sizes_off, delimiter=",", fmt="%s")
        np.savetxt(full_path_new_off, sims_new_off, delimiter=",", fmt="%s")
        np.savetxt(full_path_survived_off, sims_survived_off, delimiter=",", fmt="%s")
