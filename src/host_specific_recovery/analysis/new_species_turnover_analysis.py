import numpy as np
from src.host_specific_recovery.utils.general_utils import subset

def run_new_species_turnover(dataset: dict, control: bool = False) -> dict[str, np.ndarray]:

    if control:
        baseline = dataset['baseline_control']
        abx = dataset['abx_control']
        post_abx_cohorts = dataset['post_abx_cohorts_control']
    else:
        baseline = dataset['baseline']
        abx = dataset['abx']
        post_abx_cohorts = dataset['post_abx_cohorts']

    times_post_abx = np.vstack([np.array([0]), np.array(dataset['times_post_abx'])[:, np.newaxis]])

    delta_times_post_abx = np.diff(times_post_abx, axis=0)

    new_species_counts_all = []

    for i, (abx, base) in enumerate(zip(abx, baseline)):

        post_matrix = np.vstack([post_abx_cohorts[j][i] for j in range(len(post_abx_cohorts))])

        new_species = subset(post_matrix, base, abx, strict=True, new=True)
        new_species_counts = np.array(new_species).sum(axis=1)
        new_species_counts_all.append(new_species_counts)

    new_species_counts_all = np.array(new_species_counts_all)
    new_species_counts_all_rate = np.divide(new_species_counts_all[:, :-1], delta_times_post_abx[0:-1].T)

    return {
        "new_species_counts_all_rate": new_species_counts_all_rate.T,
        "times": times_post_abx[1:-1],
    }
