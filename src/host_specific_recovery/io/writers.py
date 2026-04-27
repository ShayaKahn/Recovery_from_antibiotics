from pathlib import Path
import numpy as np
import csv
import os
import pickle

def save_sda_results(outputs: dict, base_dir: str | Path) -> None:

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

    np.save(base_dir / "ranks.npy", np.array(outputs["ranks"]))
    np.save(base_dir / "ranks_mid.npy", np.array(outputs["ranks_mid"]))
    np.save(base_dir / "ranks_naive.npy", np.array(outputs["ranks_naive"]))


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

    np.savetxt(full_path_new, outputs["sim_new_mat_numpy"], delimiter=",", fmt="%s")
    np.savetxt(full_path_others, outputs["sim_others_mat_numpy"], delimiter=",", fmt="%s")
    np.savetxt(full_path_sizes, outputs["sizes_mat_numpy"], delimiter=",", fmt="%s")

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

def save_functional_data_pipeline_analysis_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "AUC_mean_colon_transient_vs_lost_valid.npy", outputs["AUC_mean_colon_transient_vs_lost_valid"])
    np.save(base_dir / "adjusted_pvalues_mean_colon_transient_vs_lost_valid.npy",
            outputs["adjusted_pvalues_mean_colon_transient_vs_lost_valid"])
    np.save(base_dir / "AUC_mean_colon_transient_vs_base_valid.npy", outputs["AUC_mean_colon_transient_vs_base_valid"])
    np.save(base_dir / "adjusted_pvalues_mean_colon_transient_vs_base_valid.npy",
            outputs["adjusted_pvalues_mean_colon_transient_vs_base_valid"])

    with open(base_dir / "S_lost_colon.pkl", 'wb') as f:
        pickle.dump(outputs["S_lost_colon"], f)

    with open(base_dir / "S_lost_transient_comb.pkl", 'wb') as f:
        pickle.dump(outputs["S_lost_transient_comb"], f)

    with open(base_dir / "S_base_colon.pkl", 'wb') as f:
        pickle.dump(outputs["S_base_colon"], f)

    with open(base_dir / "S_base_transient_comb.pkl", 'wb') as f:
        pickle.dump(outputs["S_base_transient_comb"], f)

def save_species_proportions_heatmap_analysis_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "weighted_proportions.npy", outputs["weighted_proportions"])
    np.save(base_dir / "proportions.npy", outputs["proportions"])

def save_survived_species_analysis_results(outputs: dict, base_dir: str | Path) -> None:

    np.save(base_dir / "results_matrix.npy", outputs["results_matrix"])
    np.save(base_dir / "mean.npy", outputs["mean"])
