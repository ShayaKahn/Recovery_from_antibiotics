from src.host_specific_recovery.data_processing.functional_data_pipeline import FunctionalDataPipeline
from src.host_specific_recovery.utils.functional_data_pipeline_utils import (similarity_matrix, find_colonizers,
                                                                             two_sided_pvalues_bh_with_effect,
                                                                             filter_valid_pairs)
from pathlib import Path
import pandas as pd
import numpy as np

def run_functional_data_pipeline_prepare(data, rename_map, PATH_contrib_path, PATH, load_data_outputs):
    def get_subject_columns(baseline_cols, abx_cols, post_abx_cols):
        """Generate subject columns for each baseline."""
        subjects = []
        for j, baseline_col in enumerate(baseline_cols):
            abx_col = [abx[j] for abx in abx_cols]
            post_abx_col = [post[j] for post in post_abx_cols]
            subjects.append([baseline_col] + abx_col + post_abx_col)
        return subjects

    def find_taxa_categories(FDP_PATH, subject, n_post):
        """Find taxa categories for a given subject."""
        base_taxa = FDP_PATH.find_baseline_taxa(subject[0:2])
        new_taxa_dict = {
            "new_taxa": FDP_PATH.find_new_taxa(subject[0:2], [subject[2]], subject[-1])
        }
        for i in range(1, n_post + 1):
            new_taxa_dict[f"new_taxa_prev_{i}"] = FDP_PATH.find_new_taxa(subject[0:2], [subject[2]], subject[-i])
        lost_taxa = FDP_PATH.find_lost_taxa(base_cols=subject[0:2], abx_cols=[subject[2]], post_cols=subject[3:])
        base_no_loss_taxa = list(set(base_taxa) - set(lost_taxa))
        return base_taxa, new_taxa_dict, lost_taxa, base_no_loss_taxa

    def process_samples(FDP_PATH, subject, n_post):
        """Process baseline and post-antibiotic samples."""
        processed_base = FDP_PATH.subsample_stratified_fun(subject[0])
        processed_post = {"post": FDP_PATH.subsample_stratified_fun(subject[-1])}
        for i in range(1, n_post + 1):
            processed_post[f"post_prev_{i}"] = FDP_PATH.subsample_stratified_fun(subject[-i])
        return processed_base, processed_post

    def create_taxa_function_matrices(FDP_PATH, processed_base, processed_post, n_post):
        """Create taxa-function matrices for baseline and post samples."""
        base_dict = FDP_PATH.taxa_function_dict(processed_base)
        post_dicts = {key: FDP_PATH.taxa_function_dict(val) for key, val in processed_post.items()}
        base_matrix = FDP_PATH.taxa_dict_to_matrix(base_dict, fill_value=0.0)
        post_matrices = {key: FDP_PATH.taxa_dict_to_matrix(val, fill_value=0.0) for key, val in post_dicts.items()}
        return base_matrix, post_matrices

    def filter_taxa_matrices(base_matrix, post_matrices, base_no_loss_taxa, lost_taxa, new_taxa_dict, n_post):
        """Filter matrices to include only relevant taxa."""
        base_taxa = set(base_matrix.columns)
        base_no_loss = list(base_taxa.intersection(base_no_loss_taxa))
        lost_taxa_filtered = list(base_taxa.intersection(lost_taxa))
        post_taxa_filtered = {
            key: list(set(post_matrices[key].columns).intersection(new_taxa_dict[key]))
            for key in post_matrices
        }
        return (
            base_matrix.loc[:, base_no_loss],
            base_matrix.loc[:, lost_taxa_filtered],
            {key: post_matrices[key].loc[:, taxa] for key, taxa in post_taxa_filtered.items()},
        )

    # Main function logic
    FDP_PATH = FunctionalDataPipeline(
        fun_by_sample_table=PATH, rename_map=rename_map,
        stratified_fun=Path(PATH_contrib_path), ASV_table=data
    )

    baseline_cols = load_data_outputs['baseline'].columns.tolist()
    abx_cols = [abx.columns.tolist() for abx in load_data_outputs['abx_cohorts']]
    post_abx_cols = [post_abx.columns.tolist() for post_abx in load_data_outputs['post_abx_cohorts']]
    n_post = len(post_abx_cols)

    subjects_cols = get_subject_columns(baseline_cols, abx_cols, post_abx_cols)

    M_base_no_lost_taxa_lst_PATH = []
    M_lost_taxa_lst_PATH = []
    M_new_taxa_lst_PATH = []
    M_new_taxa_lst_PATH_dict = {f"M_new_taxa_lst_prev_{i}_PATH": [] for i in range(1, n_post + 1)}

    for subject in subjects_cols:
        base_taxa, new_taxa_dict, lost_taxa, base_no_loss_taxa = find_taxa_categories(FDP_PATH, subject, n_post)
        processed_base, processed_post = process_samples(FDP_PATH, subject, n_post)
        base_matrix, post_matrices = create_taxa_function_matrices(FDP_PATH, processed_base, processed_post, n_post)
        base_no_loss_matrix, lost_taxa_matrix, new_taxa_matrices = filter_taxa_matrices(
            base_matrix, post_matrices, base_no_loss_taxa, lost_taxa, new_taxa_dict, n_post
        )

        # Store results
        M_base_no_lost_taxa_lst_PATH.append(base_no_loss_matrix)
        M_lost_taxa_lst_PATH.append(lost_taxa_matrix)
        M_new_taxa_lst_PATH.append(new_taxa_matrices["post"])
        for i in range(1, n_post + 1):
            M_new_taxa_lst_PATH_dict[f"M_new_taxa_lst_prev_{i}_PATH"].append(new_taxa_matrices[f"post_prev_{i}"])

    return {
        "M_base_no_lost_taxa_lst_PATH": M_base_no_lost_taxa_lst_PATH,
        "M_lost_taxa_lst_PATH": M_lost_taxa_lst_PATH,
        "M_new_taxa_lst_PATH": M_new_taxa_lst_PATH,
        "M_new_taxa_lst_PATH_dict": M_new_taxa_lst_PATH_dict,
    }

def run_functional_data_pipeline_analysis(prepare_outputs: dict,  load_data_outputs: dict,
                                          metric: str = "weighted_jaccard", min_n: int = 8,  alpha: float = 0.05,
                                          decimals: int = 6) -> dict:
    """
    Analyze functional data pipeline outputs to compute similarity matrices and statistical tests.

    :param prepare_outputs: Preprocessed outputs from the functional data pipeline.
    :param load_data_outputs: Loaded data outputs containing baseline, antibiotics, and post-antibiotics cohorts.
    :param metric: Similarity metric to use (default: "weighted_jaccard").
    :param min_n: Minimum sample size for statistical tests.
    :param alpha: Significance level for Benjamini-Hochberg correction.
    :param decimals: Number of decimals for p-value formatting.
    :return: Dictionary containing valid AUCs and adjusted p-values for colonizers and transient taxa.
    """
    def compute_similarity_matrices(base_matrix, new_matrix, prev_matrices, metric):
        """Compute similarity matrices for given base and new matrices."""
        similarity = similarity_matrix(base_matrix, new_matrix, metric)
        prev_similarities = {
            f"prev_{i}": similarity_matrix(base_matrix, prev_matrices[f"prev_{i}"], metric)
            for i in range(1, len(prev_matrices) + 1)
        }
        return similarity, prev_similarities

    def extract_transient_data(base, abx_lst, post_lst, n_post):
        """Extract colonizers and transient taxa."""
        colonizers, transient = find_colonizers(base, abx_lst, post_lst)
        transient_prev = {
            f"prev_{i}": find_colonizers(base, abx_lst, post_lst, k=i)[1]
            for i in range(3, n_post + 2)
        }
        return colonizers, transient, transient_prev

    # Extract matrices and initialize variables
    M_base_no_lost_taxa_lst = prepare_outputs["M_base_no_lost_taxa_lst_PATH"]
    M_lost_taxa_lst = prepare_outputs["M_lost_taxa_lst_PATH"]
    M_new_taxa_lst = prepare_outputs["M_new_taxa_lst_PATH"]
    M_new_taxa_prev_dict = prepare_outputs["M_new_taxa_lst_PATH_dict"]
    n_post = len(M_new_taxa_prev_dict)

    mean_pairs, mean_pairs_lost, mean_pairs_base = [], [], []

    S_lost_colon, S_lost_transient_comb, S_base_colon, S_base_transient_comb = None, None, None, None

    for idx in range(len(M_lost_taxa_lst)):
        # Extract matrices for the current subject
        M_lost_taxa = M_lost_taxa_lst[idx]
        M_new_taxa = M_new_taxa_lst[idx]
        M_base_no_lost_taxa = M_base_no_lost_taxa_lst[idx]
        M_new_taxa_prev = {
            f"prev_{i}": M_new_taxa_prev_dict[f"M_new_taxa_lst_prev_{i}_PATH"][idx]
            for i in range(1, n_post + 1)
        }

        # Compute similarity matrices
        S_lost_new, S_lost_new_prev = compute_similarity_matrices(M_lost_taxa, M_new_taxa, M_new_taxa_prev, metric)
        S_base_new, S_base_new_prev = compute_similarity_matrices(M_base_no_lost_taxa, M_new_taxa, M_new_taxa_prev, metric)

        # Extract colonizers and transient taxa
        base = load_data_outputs["baseline"].columns[idx]
        abx_lst = [abx.columns[idx] for abx in load_data_outputs["abx_cohorts"]]
        post_lst = [post_abx.columns[idx] for post_abx in load_data_outputs["post_abx_cohorts"]]
        colonizers, transient, transient_prev = extract_transient_data(base, abx_lst, post_lst, n_post)

        # Filter similarity matrices for colonizers and transient taxa
        S_lost_colon = S_lost_new.loc[S_lost_new.index.intersection(colonizers)]
        S_base_colon = S_base_new.loc[S_base_new.index.intersection(colonizers)]

        S_lost_transient_comb = pd.concat(
            [S_lost_new_prev[f"prev_{i}"].loc[S_lost_new_prev[f"prev_{i}"].index.intersection(transient_prev[f"prev_{i}"])]
             for i in range(1, n_post + 1)], axis=0
        )
        S_base_transient_comb = pd.concat(
            [S_base_new_prev[f"prev_{i}"].loc[S_base_new_prev[f"prev_{i}"].index.intersection(transient_prev[f"prev_{i}"])]
             for i in range(1, n_post + 1)], axis=0
        )

        # Combine and compute means
        mean_pairs.append((
            pd.concat([S_lost_colon, S_base_colon], axis=1).mean(axis=1).to_numpy(),
            pd.concat([S_lost_transient_comb, S_base_transient_comb], axis=1).mean(axis=1).to_numpy()
        ))
        mean_pairs_lost.append((S_lost_colon.mean(axis=1).to_numpy(), S_lost_transient_comb.mean(axis=1).to_numpy()))
        mean_pairs_base.append((S_base_colon.mean(axis=1).to_numpy(), S_base_transient_comb.mean(axis=1).to_numpy()))

    # Perform statistical tests
    results_lost = two_sided_pvalues_bh_with_effect(mean_pairs_lost, alpha=alpha, decimals=decimals, min_n=min_n)
    results_base = two_sided_pvalues_bh_with_effect(mean_pairs_base, alpha=alpha, decimals=decimals, min_n=min_n)

    # Extract valid results
    AUC_lost, pvals_lost = np.array([res["auc"] for res in results_lost]), np.array([res["q_bh"] for res in results_lost])
    AUC_base, pvals_base = np.array([res["auc"] for res in results_base]), np.array([res["q_bh"] for res in results_base])

    AUC_lost_valid, pvals_lost_valid, AUC_base_valid, pvals_base_valid = filter_valid_pairs(
        AUC_lost, pvals_lost, AUC_base, pvals_base
    )

    return {
        "AUC_mean_colon_transient_vs_lost_valid": AUC_lost_valid,
        "adjusted_pvalues_mean_colon_transient_vs_lost_valid": pvals_lost_valid,
        "AUC_mean_colon_transient_vs_base_valid": AUC_base_valid,
        "adjusted_pvalues_mean_colon_transient_vs_base_valid": pvals_base_valid,
        "S_lost_colon": S_lost_colon,
        "S_lost_transient_comb": S_lost_transient_comb,
        "S_base_colon": S_base_colon,
        "S_base_transient_comb": S_base_transient_comb,
    }
