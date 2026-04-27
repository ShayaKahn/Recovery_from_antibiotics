from src.host_specific_recovery.statistical_models.surrogate import Surrogate
from src.host_specific_recovery.utils.surrogate_data_analysis_utils import create_cohort_dict
import numpy as np
from scipy.stats import binom, binomtest

def run_surrogate_analysis(dataset: dict, timepoints_val:int,  method: str = "Jaccard") -> dict:
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
        sim_others_mid.append([res[key] for key in keys if key != specific_key])
        sim_others_naive.append([res[key] for key in keys if key != specific_key])

    # calculate the ranks
    ranks = np.array([len(keys) - np.sum((sim_val > sim_val_others)
                                         ) for sim_val, sim_val_others in zip(sim, sim_others)])
    ranks_mid = np.array([len(keys) - np.sum((sim_val > sim_val_others)
                                             ) for sim_val, sim_val_others in zip(sim_mid, sim_others_mid)])
    ranks_naive = np.array([len(keys) - np.sum((sim_val > sim_val_others)
                                               ) for sim_val, sim_val_others in zip(sim_naive, sim_others_naive)])

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
        "ranks_naive": ranks_naive
    }

def run_binomial_test(surrogate_outputs, alpha=0.9):

    n_subjects = len(surrogate_outputs["ranks"])
    n_success = (surrogate_outputs["ranks"] == 1).sum()
    prob_null = 1 / (len(surrogate_outputs["sim_others"]) + 1)

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
        "interval_null": interval_null
    }
