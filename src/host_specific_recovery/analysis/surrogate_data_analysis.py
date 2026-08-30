from src.host_specific_recovery.utils.surrogate_data_analysis_utils import create_cohort_dict
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, norm
from statsmodels.stats.multitest import multipletests


def calculate_sda_statistics(
        observed_similarities,
        reference_similarities,
        subject_ids=None,
        alpha=0.05,
) -> pd.DataFrame:
    """Calculate subject-level SDA statistics from reference similarities.

    The definitions match ``plot_SDA``: reference standard deviation uses
    ``ddof=0``; p-values are two-sided normal probabilities and are adjusted
    across subjects with Benjamini-Hochberg. A result is significant only when
    its adjusted p-value is at most ``alpha`` and its observed similarity is
    greater than every reference similarity.

    Raw rank uses 1 as the best possible rank. ``normalized_rank`` uses 1 for
    best and 0 for worst, which makes ranks comparable across reference-pool
    sizes. A zero reference standard deviation produces an undefined (NaN)
    standardized similarity and a non-significant result.
    """
    observed_similarities = np.asarray(observed_similarities, dtype=float)
    if observed_similarities.ndim == 0:
        observed_similarities = observed_similarities.reshape(1)
    if observed_similarities.ndim != 1:
        raise ValueError("observed_similarities must be a one-dimensional sequence.")

    reference_similarities = list(reference_similarities)
    if len(reference_similarities) != observed_similarities.size:
        raise ValueError(
            "reference_similarities must contain one sequence per observed similarity."
        )
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number in the interval (0, 1).")

    if subject_ids is None:
        subject_ids = np.arange(observed_similarities.size)
    else:
        subject_ids = np.asarray(list(subject_ids), dtype=object)
        if subject_ids.ndim != 1 or subject_ids.size != observed_similarities.size:
            raise ValueError("subject_ids must contain one identifier per subject.")

    rows = []
    for subject_id, observed, references in zip(
            subject_ids,
            observed_similarities,
            reference_similarities,
    ):
        references = np.asarray(references, dtype=float)
        if references.ndim != 1 or references.size == 0:
            raise ValueError(
                f"Reference similarities for subject {subject_id!r} must be a non-empty "
                "one-dimensional sequence."
            )
        if not np.isfinite(observed) or not np.all(np.isfinite(references)):
            raise ValueError(
                f"Similarities for subject {subject_id!r} must contain only finite values."
            )

        reference_mean = float(references.mean())
        reference_std = float(references.std(ddof=0))
        if reference_std == 0:
            standardized_similarity = np.nan
            p_value = np.nan
        else:
            standardized_similarity = float(
                (observed - reference_mean) / reference_std
            )
            p_value = float(
                2 * (1 - norm.cdf(abs(standardized_similarity)))
            )

        n_references = int(references.size)
        n_references_below_observed = int(np.count_nonzero(observed > references))
        rank = n_references + 1 - n_references_below_observed

        rows.append({
            "subject_id": subject_id,
            "observed_similarity": float(observed),
            "reference_mean_similarity": reference_mean,
            "reference_std_similarity": reference_std,
            "n_references": n_references,
            "rank": int(rank),
            "normalized_rank": float(
                n_references_below_observed / n_references
            ),
            "standardized_similarity": standardized_similarity,
            "p_value": p_value,
            "observed_exceeds_all_references": bool(
                observed > references.max()
            ),
        })

    statistics = pd.DataFrame(rows)
    statistics["adjusted_p_value"] = np.nan
    finite_p_values = np.isfinite(statistics["p_value"].to_numpy(dtype=float))
    if np.any(finite_p_values):
        statistics.loc[finite_p_values, "adjusted_p_value"] = multipletests(
            statistics.loc[finite_p_values, "p_value"].to_numpy(dtype=float),
            alpha=alpha,
            method="fdr_bh",
        )[1]
    statistics["is_significant"] = (
        (statistics["adjusted_p_value"] <= alpha)
        & statistics["observed_exceeds_all_references"]
    )
    return statistics

def run_surrogate_analysis(dataset: dict, timepoints_val:int,  method: str = "Jaccard") -> dict:
    from src.host_specific_recovery.statistical_models.surrogate import Surrogate

    # initialize results
    results = []
    results_obj = []
    results_mid = []
    results_obj_mid = []
    results_naive = []
    results_obj_naive = []

    baseline = dataset["baseline"]
    baseline_full = dataset["baseline_full"]
    post_abx_cohorts = dataset["post_abx_cohorts"]
    abx = dataset["abx"]
    keys = dataset["keys"]
    filtered_keys = dataset["filtered_keys"]

    for i, key in enumerate(filtered_keys):
        subject_base = {key: baseline[i, :]}
        subject_base_dict = create_cohort_dict(keys, key, baseline_full)
        subject_abx = abx[i, :]

        test_post_abx_matrix = np.vstack([post_abx_cohorts[tp][i, :] for tp in range(len(post_abx_cohorts))])

        # apply surrogate data analysis
        results_subject_obj = Surrogate(subject_base_dict, subject_base, test_post_abx_matrix, subject_abx,
                                        timepoints=timepoints_val, strict=False)
        results_subject = results_subject_obj.apply_surrogate_data_analysis(method=method)
        results.append(results_subject)
        results_obj.append(results_subject_obj)

        # Remove only survived species
        results_subject_obj_mid = Surrogate(subject_base_dict, subject_base, test_post_abx_matrix, subject_abx,
                                            timepoints=0, strict=False)
        results_subject_mid = results_subject_obj_mid.apply_surrogate_data_analysis(method=method)
        results_mid.append(results_subject_mid)
        results_obj_mid.append(results_subject_obj_mid)

        # Naive method
        results_subject_obj_naive = Surrogate(subject_base_dict, subject_base, test_post_abx_matrix, subject_abx,
                                              timepoints=0, strict=False, naive=True)
        results_subject_naive = results_subject_obj_naive.apply_surrogate_data_analysis(method=method)
        results_naive.append(results_subject_naive)
        results_obj_naive.append(results_subject_obj_naive)

    sim = []
    sim_mid = []
    sim_naive = []
    sim_others = []
    sim_others_mid = []
    sim_others_naive = []
    for specific_key, res, res_mid, res_naive in zip(filtered_keys, results, results_mid, results_naive):
        sim.append(res[specific_key])
        sim_mid.append(res_mid[specific_key])
        sim_naive.append(res_naive[specific_key])
        sim_others.append([res[key] for key in keys if key != specific_key])
        sim_others_mid.append([res_mid[key] for key in keys if key != specific_key])
        sim_others_naive.append([res_naive[key] for key in keys if key != specific_key])

    statistics = calculate_sda_statistics(sim, sim_others, filtered_keys)
    statistics_mid = calculate_sda_statistics(sim_mid, sim_others_mid, filtered_keys)
    statistics_naive = calculate_sda_statistics(
        sim_naive,
        sim_others_naive,
        filtered_keys,
    )
    ranks = statistics["rank"].to_numpy(dtype=int)
    ranks_mid = statistics_mid["rank"].to_numpy(dtype=int)
    ranks_naive = statistics_naive["rank"].to_numpy(dtype=int)

    return {
        "results": results,
        "results_obj": results_obj,
        "results_mid": results_mid,
        "results_obj_mid": results_obj_mid,
        "results_naive": results_naive,
        "results_obj_naive": results_obj_naive,
        "similarity": sim,
        "similarity_others": sim_others,
        "similarity_mid": sim_mid,
        "similarity_others_mid": sim_others_mid,
        "similarity_naive": sim_naive,
        "similarity_others_naive": sim_others_naive,
        "ranks": ranks,
        "ranks_mid": ranks_mid,
        "ranks_naive": ranks_naive,
        "statistics": statistics,
        "statistics_mid": statistics_mid,
        "statistics_naive": statistics_naive,
    }

def run_binomial_test(surrogate_outputs, alpha=0.9):

    n_subjects = len(surrogate_outputs["ranks"])
    n_success = (surrogate_outputs["ranks"] == 1).sum()
    prob_null = 1 / (len(surrogate_outputs["similarity_others"][0]) + 1)

    def binomial_test(n, k, p):
        result = binomtest(k, n, p, alternative='greater')
        return result.pvalue, result.proportion_ci(confidence_level=0.95, method='wilson')

    def calc_interval(n, p, alpha=alpha):
        return binom.interval(alpha, n, p)

    p, ci = binomial_test(n_subjects, n_success, prob_null)
    interval_null = calc_interval(n_subjects, prob_null)

    return {
        "p_value": p,
        "confidence_interval": ci,
        "interval_null": interval_null,
    }
