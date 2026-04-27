from src.host_specific_recovery.statistical_models.similarity_correlation import SimilarityCorrelation
import numpy as np
import pandas as pd

def run_similarity_correlation(dataset: dict, timepoints_val:int,  method: str = "Jaccard", n_jobs: int = 4) -> dict:

    filtered_keys = dataset["filtered_keys"]
    abx = dataset["abx"]
    baseline = dataset["baseline"]
    baseline_full = dataset["baseline_full"]
    keys = dataset["keys"]
    post_abx_cohorts = dataset["post_abx_cohorts"]

    abx_frame = pd.DataFrame(abx.T, columns=filtered_keys)
    baseline_frame = pd.DataFrame(baseline.T, columns=filtered_keys)
    baseline_ref_frame = pd.DataFrame(baseline_full.T, columns=keys)
    iters = None
    species_type = 'new'
    strict = True
    zscore = False
    post_abx_container = {}
    for j, key in enumerate(filtered_keys):
        post_abx_mat = np.vstack([post_abx_cohorts[tp][j, :] for tp in range(len(post_abx_cohorts))])
        post_abx_container[key] = post_abx_mat

    # Apply similarity correlation
    sim = SimilarityCorrelation(abx_frame, baseline_frame, post_abx_container, baseline_ref_frame, method,
                                timepoints_val, iters, species_type, strict, zscore=zscore, n_jobs=n_jobs)
    sims_container_shifted = sim.calc_similarity()

    # Construct results
    sizes = {}
    for key_ref in keys:
        sizes[key_ref] = np.size(np.nonzero(baseline_ref_frame[key_ref]))

    sim_new_mat = []
    sim_others_mat = []
    sizes_mat = []
    for key in filtered_keys:
        sims_n = [val[0] for val in list(sims_container_shifted[key].values())]
        sims_o = [val[1] for val in list(sims_container_shifted[key].values())]
        sizes_vals = [sizes[key_ref] for key_ref in keys if key_ref != key]
        sim_new_mat.append(sims_n)
        sim_others_mat.append(sims_o)
        sizes_mat.append(sizes_vals)

    sim_new_mat_numpy = np.array(sim_new_mat)
    sim_others_mat_numpy = np.array(sim_others_mat)
    sizes_mat_numpy = np.array(sizes_mat)

    return {
        "similarity_new": sim_new_mat_numpy,
        "similarity_others": sim_others_mat_numpy,
        "sizes": sizes_mat_numpy
    }

