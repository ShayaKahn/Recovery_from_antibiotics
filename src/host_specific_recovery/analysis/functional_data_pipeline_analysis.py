from src.host_specific_recovery.data_processing.functional_data_pipeline import FunctionalDataPipeline
from src.host_specific_recovery.metrics.similarity import Similarity
from src.host_specific_recovery.utils.functional_data_pipeline_utils import (similarity_matrix, find_colonizers,
                                                                             two_sided_pvalues_bh_with_effect,
                                                                             filter_valid_pairs)
from pathlib import Path
import pandas as pd
import numpy as np

def run_functional_data_pipeline_prepare(data, rename_map, PATH_contrib_path, PATH, load_data_outputs):
    def load_stratified_rows(FDP_PATH, sample_names):
        """Load all needed stratified rows with one DuckDB scan."""
        names = list(dict.fromkeys(sample_names))
        if not names:
            raise ValueError("No samples were provided for functional data preparation")

        if rename_map is None:
            sample_rows = FDP_PATH.con.execute(
                """
                SELECT
                    sample::VARCHAR AS sample,
                    function::VARCHAR AS function,
                    taxon::VARCHAR AS taxon,
                    genome_function_count::FLOAT AS genome_function_count
                FROM read_csv_auto(?, delim='\t', header=true)
                WHERE sample IN (SELECT * FROM UNNEST(?))
                """,
                [str(FDP_PATH.stratified_fun), names],
            ).df()
        else:
            rename_df = pd.DataFrame(
                {"run": list(rename_map.keys()), "name": list(rename_map.values())}
            )
            FDP_PATH.con.register("rename_map_df", rename_df)
            FDP_PATH.con.execute(
                "CREATE OR REPLACE TEMP TABLE rename_map AS SELECT * FROM rename_map_df"
            )
            FDP_PATH.con.unregister("rename_map_df")

            sample_rows = FDP_PATH.con.execute(
                """
                WITH stratified AS (
                    SELECT
                        COALESCE(rm.name, s.sample)::VARCHAR AS sample,
                        s.function::VARCHAR AS function,
                        s.taxon::VARCHAR AS taxon,
                        s.genome_function_count::FLOAT AS genome_function_count
                    FROM read_csv_auto(?, delim='\t', header=true) s
                    LEFT JOIN rename_map rm
                        ON s.sample = rm.run
                )
                SELECT sample, function, taxon, genome_function_count
                FROM stratified
                WHERE sample IN (SELECT * FROM UNNEST(?))
                """,
                [str(FDP_PATH.stratified_fun), names],
            ).df()

        found_samples = set(sample_rows["sample"].unique())
        missing_samples = [name for name in names if name not in found_samples]
        if missing_samples:
            raise ValueError(f"Samples not found in file {FDP_PATH.stratified_fun}: {missing_samples}")

        return {
            sample: rows.drop(columns="sample").reset_index(drop=True)
            for sample, rows in sample_rows.groupby("sample", sort=False)
        }

    def sample_rows_to_matrix(sample_rows):
        """Convert rows for one sample to the same function-by-taxon matrix as the pipeline class."""
        dup = sample_rows.duplicated(["taxon", "function"], keep=False)
        if dup.any():
            bad = sample_rows.loc[dup, ["taxon", "function"]].drop_duplicates().head(10)
            raise ValueError(f"Found duplicate (taxon, function) pairs. Examples:\n{bad}")

        taxa_fun_dict = {}
        for taxon, sub in sample_rows.groupby("taxon", sort=False):
            taxa_fun_dict[str(taxon)] = pd.to_numeric(
                sub.set_index("function")["genome_function_count"],
                errors="coerce",
            )

        matrix = pd.concat(taxa_fun_dict, axis=1)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return matrix

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
            "post": FDP_PATH.find_new_taxa(subject[0:2], [subject[2]], subject[-1])
        }
        for i in range(1, n_post):
            new_taxa_dict[f"post_prev_{i}"] = FDP_PATH.find_new_taxa(subject[0:2], [subject[2]], subject[-i - 1])
        lost_taxa = FDP_PATH.find_lost_taxa(base_cols=subject[0:2], abx_cols=[subject[2]], post_cols=subject[3:])
        base_no_loss_taxa = list(set(base_taxa) - set(lost_taxa))
        return base_taxa, new_taxa_dict, lost_taxa, base_no_loss_taxa

    def create_taxa_function_matrices(sample_matrices, subject, n_post):
        """Process sample matrices for one subject."""
        base_matrix = sample_matrices[subject[0]]
        subject_matrices = [sample_matrices[subject_col] for subject_col in subject]
        post_matrices = {"post": sample_matrices[subject[-1]]}
        for i in range(1, n_post):
            post_matrices[f"post_prev_{i}"] = sample_matrices[subject[-i - 1]]
        return base_matrix, post_matrices, subject_matrices

    def compute_taxa_functional_similarity_matrix(taxa_fun_matrix):
        """Compute weighted-Jaccard similarity between all taxa in one function-by-taxon matrix."""
        return similarity_matrix(taxa_fun_matrix, taxa_fun_matrix, metric="weighted_jaccard")

    def compute_functional_redundancy(subject_col, taxa_fun_matrix, ASV_table):

        subject_sample = ASV_table.loc[:, subject_col]
        taxa = list(taxa_fun_matrix.columns)
        sample_taxa_rel_abun = pd.to_numeric(
            subject_sample.reindex(taxa, fill_value=0.0),
            errors="coerce",
        ).fillna(0.0)

        taxa_functional_similarity = compute_taxa_functional_similarity_matrix(taxa_fun_matrix)
        taxa_functional_similarity = taxa_functional_similarity.loc[taxa, taxa]
        weights = sample_taxa_rel_abun.to_numpy(dtype=float, copy=False)
        similarities = taxa_functional_similarity.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(similarities, 0.0)

        return float(weights @ similarities @ weights)

    def functional_profile(taxa_fun_matrix):
        """Collapse a function-by-taxon matrix to one function abundance profile."""
        return taxa_fun_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    def compute_specificity_to_baseline(subject_matrices):
        baseline_profile = functional_profile(subject_matrices[0])
        specificity_values = []
        for subject_matrix in subject_matrices:
            sample_profile = functional_profile(subject_matrix)
            aligned_sample, aligned_baseline = sample_profile.align(baseline_profile, join="outer", fill_value=0.0)
            specificity_values.append(float(
                Similarity(
                    aligned_sample.to_numpy(dtype=float),
                    aligned_baseline.to_numpy(dtype=float),
                    method="Specificity",
                    norm=False,
                ).calculate_similarity()
            ))
        return specificity_values

    def filter_taxa_matrices(base_matrix, post_matrices, base_no_loss_taxa, lost_taxa, new_taxa_dict):
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
    ASV_table = data.rename(columns=rename_map).astype(float) if rename_map is not None else data
    FDP_PATH = FunctionalDataPipeline(
        fun_by_sample_table=PATH,
        rename_map=None,
        stratified_fun=Path(PATH_contrib_path),
        ASV_table=ASV_table,
    )

    baseline_cols = load_data_outputs['baseline_df'].columns.tolist()
    abx_cols = [abx.columns.tolist() for abx in load_data_outputs['abx_cohorts_df']]
    post_abx_cols = [post_abx.columns.tolist() for post_abx in load_data_outputs['post_abx_cohorts_df']]
    n_post = len(post_abx_cols)

    subjects_cols = get_subject_columns(baseline_cols, abx_cols, post_abx_cols)
    samples_to_load = []
    for subject in subjects_cols:
        samples_to_load.extend(subject)

    stratified_rows = load_stratified_rows(FDP_PATH, samples_to_load)
    sample_matrices = {
        sample: sample_rows_to_matrix(rows)
        for sample, rows in stratified_rows.items()
    }

    M_base_no_lost_taxa_lst_PATH = []
    M_lost_taxa_lst_PATH = []
    M_new_taxa_lst_PATH = []
    M_new_taxa_lst_PATH_dict = {f"M_new_taxa_lst_prev_{i}_PATH": [] for i in range(1, n_post)}
    fun_redundancy_lst = []
    specificity_similarity_lst = []

    for subject in subjects_cols:
        base_taxa, new_taxa_dict, lost_taxa, base_no_loss_taxa = find_taxa_categories(FDP_PATH, subject, n_post)
        base_matrix, post_matrices, subject_matrices = create_taxa_function_matrices(sample_matrices, subject, n_post)
        base_no_loss_matrix, lost_taxa_matrix, new_taxa_matrices = filter_taxa_matrices(
            base_matrix, post_matrices, base_no_loss_taxa, lost_taxa, new_taxa_dict
        )

        # Store results
        M_base_no_lost_taxa_lst_PATH.append(base_no_loss_matrix)
        M_lost_taxa_lst_PATH.append(lost_taxa_matrix)
        M_new_taxa_lst_PATH.append(new_taxa_matrices["post"])
        for i in range(1, n_post):
            M_new_taxa_lst_PATH_dict[f"M_new_taxa_lst_prev_{i}_PATH"].append(new_taxa_matrices[f"post_prev_{i}"])
        fun_redundancy_lst.append([
            compute_functional_redundancy(subject_col, subject_matrix, ASV_table)
            for subject_col, subject_matrix in zip(subject, subject_matrices)
        ])
        specificity_similarity_lst.append(compute_specificity_to_baseline(subject_matrices))

    return {
        "M_base_no_lost_taxa_lst_PATH": M_base_no_lost_taxa_lst_PATH,
        "M_lost_taxa_lst_PATH": M_lost_taxa_lst_PATH,
        "M_new_taxa_lst_PATH": M_new_taxa_lst_PATH,
        "M_new_taxa_lst_PATH_dict": M_new_taxa_lst_PATH_dict,
        "fun_redundancy_lst": fun_redundancy_lst,
        "specificity_similarity_lst": specificity_similarity_lst,
    }

def run_functional_data_pipeline_analysis(prepare_outputs: dict,  load_data_outputs: dict,
                                          metric: str = "weighted_jaccard", min_n: int = 8,  alpha: float = 0.05,
                                          decimals: int = 6, subject_idx: int = -1,
                                          only_last: bool = False) -> dict:
    """
    Analyze functional data pipeline outputs to compute similarity matrices and statistical tests.

    :param prepare_outputs: Preprocessed outputs from the functional data pipeline.
    :param load_data_outputs: Loaded data outputs containing baseline, antibiotics, and post-antibiotics cohorts.
    :param metric: Similarity metric to use (default: "weighted_jaccard").
    :param min_n: Minimum sample size for statistical tests.
    :param alpha: Significance level for Benjamini-Hochberg correction.
    :param decimals: Number of decimals for p-value formatting.
    :param subject_idx: Subject index whose similarity matrices should be returned. Defaults to -1, the last subject.
    :param only_last: If True, use only the prev_1 transient matrices instead of concatenating all prev_i matrices.
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

    def extract_transient_data(base, abx_lst, post_lst, n_prev):
        """Extract colonizers and transient taxa."""
        colonizers, transient = find_colonizers(base, abx_lst, post_lst)
        transient_prev = {
            f"prev_{i}": find_colonizers(base, abx_lst, post_lst, k=i + 1)[1]
            for i in range(1, n_prev + 1)
        }
        return colonizers, transient, transient_prev

    # Extract matrices and initialize variables
    M_base_no_lost_taxa_lst = prepare_outputs["M_base_no_lost_taxa_lst_PATH"]
    M_lost_taxa_lst = prepare_outputs["M_lost_taxa_lst_PATH"]
    M_new_taxa_lst = prepare_outputs["M_new_taxa_lst_PATH"]
    M_new_taxa_prev_dict = prepare_outputs["M_new_taxa_lst_PATH_dict"]
    n_prev = len(M_new_taxa_prev_dict)
    n_subjects = len(M_lost_taxa_lst)
    selected_subject_idx = subject_idx if subject_idx >= 0 else n_subjects + subject_idx
    if selected_subject_idx < 0 or selected_subject_idx >= n_subjects:
        raise IndexError(f"subject_idx={subject_idx} is out of range for {n_subjects} subjects")

    mean_pairs, mean_pairs_lost, mean_pairs_base = [], [], []
    mean_S_lost_base = []

    S_lost_colon_selected = None
    S_lost_transient_comb_selected = None
    S_lost_base_selected = None
    S_base_colon_selected = None
    S_base_transient_comb_selected = None

    for idx in range(len(M_lost_taxa_lst)):
        # Extract matrices for the current subject
        M_lost_taxa = M_lost_taxa_lst[idx]
        M_new_taxa = M_new_taxa_lst[idx]
        M_base_no_lost_taxa = M_base_no_lost_taxa_lst[idx]
        M_new_taxa_prev = {
            f"prev_{i}": M_new_taxa_prev_dict[f"M_new_taxa_lst_prev_{i}_PATH"][idx]
            for i in range(1, n_prev + 1)
        }

        # Compute similarity matrices
        S_lost_new, S_lost_new_prev = compute_similarity_matrices(M_lost_taxa, M_new_taxa, M_new_taxa_prev, metric)
        S_base_new, S_base_new_prev = compute_similarity_matrices(M_base_no_lost_taxa, M_new_taxa, M_new_taxa_prev, metric)
        S_lost_base = similarity_matrix(M_lost_taxa, M_base_no_lost_taxa, metric)

        # Extract colonizers and transient taxa
        base = [
            load_data_outputs["baseline_df"].iloc[:, idx],
            load_data_outputs["abx_cohorts_df"][0].iloc[:, idx],
        ]
        abx_lst = [abx.iloc[:, idx] for abx in load_data_outputs["abx_cohorts_df"][1:]]
        post_lst = [post_abx.iloc[:, idx] for post_abx in load_data_outputs["post_abx_cohorts_df"]]
        colonizers, transient, transient_prev = extract_transient_data(base, abx_lst, post_lst, n_prev)

        # Filter similarity matrices for colonizers and transient taxa
        S_lost_colon = S_lost_new.loc[S_lost_new.index.intersection(colonizers)]
        S_base_colon = S_base_new.loc[S_base_new.index.intersection(colonizers)]

        if only_last:
            if n_prev < 1:
                raise ValueError("only_last=True requires at least one previous post-antibiotic cohort.")
            S_lost_transient_comb = S_lost_new_prev["prev_1"].loc[
                S_lost_new_prev["prev_1"].index.intersection(transient_prev["prev_1"])
            ]
            S_base_transient_comb = S_base_new_prev["prev_1"].loc[
                S_base_new_prev["prev_1"].index.intersection(transient_prev["prev_1"])
            ]
        else:
            S_lost_transient_comb = pd.concat(
                [S_lost_new_prev[f"prev_{i}"].loc[S_lost_new_prev[f"prev_{i}"].index.intersection(transient_prev[f"prev_{i}"])]
                 for i in range(1, n_prev + 1)], axis=0)

            S_base_transient_comb = pd.concat(
                [S_base_new_prev[f"prev_{i}"].loc[S_base_new_prev[f"prev_{i}"].index.intersection(transient_prev[f"prev_{i}"])]
                 for i in range(1, n_prev + 1)], axis=0)

        if idx == selected_subject_idx:
            S_lost_colon_selected = S_lost_colon
            S_lost_transient_comb_selected = S_lost_transient_comb
            S_base_colon_selected = S_base_colon
            S_base_transient_comb_selected = S_base_transient_comb
            S_lost_base_selected = S_lost_base

        # Combine and compute means
        mean_pairs.append((
            pd.concat([S_lost_colon, S_base_colon], axis=1).mean(axis=1).to_numpy(),
            pd.concat([S_lost_transient_comb, S_base_transient_comb], axis=1).mean(axis=1).to_numpy()
        ))
        mean_pairs_lost.append((S_lost_colon.mean(axis=1).to_numpy(), S_lost_transient_comb.mean(axis=1).to_numpy()))
        mean_pairs_base.append((S_base_colon.mean(axis=1).to_numpy(), S_base_transient_comb.mean(axis=1).to_numpy()))
        mean_S_lost_base.append(S_lost_base.to_numpy(dtype=float, copy=False).mean())

    # Perform statistical tests
    results_lost = two_sided_pvalues_bh_with_effect(mean_pairs_lost, alpha=alpha, decimals=decimals, min_n=min_n)
    results_base = two_sided_pvalues_bh_with_effect(mean_pairs_base, alpha=alpha, decimals=decimals, min_n=min_n)

    # Extract valid results
    AUC_lost, pvals_lost = np.array([res["auc"] for res in results_lost]), np.array([res["q_bh"] for res in results_lost])
    AUC_base, pvals_base = np.array([res["auc"] for res in results_base]), np.array([res["q_bh"] for res in results_base])
    mean_S_lost_base = np.asarray(mean_S_lost_base, dtype=float)
    fun_redundancy = np.asarray(prepare_outputs["fun_redundancy_lst"], dtype=float)
    if fun_redundancy.ndim == 1:
        fun_redundancy = fun_redundancy[:, np.newaxis]
    if fun_redundancy.ndim != 2 or fun_redundancy.shape[0] != AUC_lost.size:
        raise ValueError(
            "prepare_outputs['fun_redundancy_lst'] must have one row per subject "
            f"({fun_redundancy.shape[0] if fun_redundancy.ndim else 0} rows for {AUC_lost.size} subjects)."
        )
    fun_redundancy_columns = [f"sample_{i}" for i in range(fun_redundancy.shape[1])]

    valid_mask = (
            np.isfinite(AUC_lost)
            & np.isfinite(pvals_lost)
            & np.isfinite(AUC_base)
            & np.isfinite(pvals_base)
            & (0.0 <= pvals_lost)
            & (pvals_lost <= 1.0)
            & (0.0 <= pvals_base)
            & (pvals_base <= 1.0)
    )
    AUC_lost_valid, pvals_lost_valid, AUC_base_valid, pvals_base_valid, mean_S_lost_base_valid = (
        AUC_lost[valid_mask],
        pvals_lost[valid_mask],
        AUC_base[valid_mask],
        pvals_base[valid_mask],
        mean_S_lost_base[valid_mask],
    )
    fun_redundancy_valid = fun_redundancy[valid_mask]
    valid_subject_ids = np.array(load_data_outputs["filtered_keys"])[valid_mask]

    AUC_lost_valid_with_subjects = pd.Series(
        AUC_lost_valid,
        index=valid_subject_ids,
        name="AUC_mean_colon_transient_vs_lost_valid",
    )
    pvals_lost_valid_with_subjects = pd.Series(
        pvals_lost_valid,
        index=valid_subject_ids,
        name="adjusted_pvalues_mean_colon_transient_vs_lost_valid",
    )
    AUC_base_valid_with_subjects = pd.Series(
        AUC_base_valid,
        index=valid_subject_ids,
        name="AUC_mean_colon_transient_vs_base_valid",
    )
    pvals_base_valid_with_subjects = pd.Series(
        pvals_base_valid,
        index=valid_subject_ids,
        name="adjusted_pvalues_mean_colon_transient_vs_base_valid",
    )
    fun_redundancy_valid_with_subjects = pd.DataFrame(
        fun_redundancy_valid,
        index=valid_subject_ids,
        columns=fun_redundancy_columns,
    )
    fun_redundancy_valid_with_subjects.index.name = "subject"
    mean_S_lost_base_valid_with_subjects = pd.Series(
        mean_S_lost_base_valid,
        index=valid_subject_ids,
        name="mean_S_lost_base_valid",
    )
    valid_subject_results = pd.DataFrame(
        {
            "AUC_mean_colon_transient_vs_lost_valid": AUC_lost_valid,
            "adjusted_pvalues_mean_colon_transient_vs_lost_valid": pvals_lost_valid,
            "AUC_mean_colon_transient_vs_base_valid": AUC_base_valid,
            "adjusted_pvalues_mean_colon_transient_vs_base_valid": pvals_base_valid,
            "mean_S_lost_base_valid": mean_S_lost_base_valid,
        },
        index=valid_subject_ids,
    )
    valid_subject_results = pd.concat(
        [
            valid_subject_results,
            fun_redundancy_valid_with_subjects.add_prefix("fun_redundancy_"),
        ],
        axis=1,
    )
    valid_subject_results.index.name = "subject"

    return {
        "AUC_mean_colon_transient_vs_lost_valid": AUC_lost_valid,
        "adjusted_pvalues_mean_colon_transient_vs_lost_valid": pvals_lost_valid,
        "AUC_mean_colon_transient_vs_base_valid": AUC_base_valid,
        "adjusted_pvalues_mean_colon_transient_vs_base_valid": pvals_base_valid,
        "AUC_mean_colon_transient_vs_lost_valid_subjects": AUC_lost_valid_with_subjects,
        "adjusted_pvalues_mean_colon_transient_vs_lost_valid_subjects": pvals_lost_valid_with_subjects,
        "AUC_mean_colon_transient_vs_base_valid_subjects": AUC_base_valid_with_subjects,
        "adjusted_pvalues_mean_colon_transient_vs_base_valid_subjects": pvals_base_valid_with_subjects,
        "fun_redundancy_valid": fun_redundancy_valid,
        "fun_redundancy_valid_subjects": fun_redundancy_valid_with_subjects,
        "mean_S_lost_base_valid": mean_S_lost_base_valid,
        "mean_S_lost_base_valid_subjects": mean_S_lost_base_valid_with_subjects,
        "valid_subject_results": valid_subject_results,
        "S_lost_colon": S_lost_colon_selected,
        "S_lost_transient_comb": S_lost_transient_comb_selected,
        "S_lost_base": S_lost_base_selected,
        "S_base_colon": S_base_colon_selected,
        "S_base_transient_comb": S_base_transient_comb_selected,
    }
