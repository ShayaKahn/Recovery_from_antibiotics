from src.host_specific_recovery.utils.general_utils import subset
from src.host_specific_recovery.statistical_models.colonization_probability import ColonizationProbability
import numpy as np

def run_cross_species_probability_analysis(dataset: dict) -> dict:

    post_ABX = dataset["post_abx_cohorts"]
    baseline = dataset["baseline"]
    ABX = dataset["abx"]

    new_probs = []
    returned_probs = []

    for i, base, abx in enumerate(zip(baseline, ABX)):

        post_matrix = np.vstack([p[i, :] for p in post_ABX])
        new_species_prev = subset(post_matrix, base, abx, True, True)[-2]
        returned_species_prev = subset(post_matrix, base, abx, False, False)[-2]

        dim = post_matrix.shape[0] - 1
        new_species = subset(post_matrix[:dim, :], base, abx, True, True)[-1]
        returned_species = subset(post_matrix[:dim, :], base, abx, False, False)[-1]

        new_probs.append(np.sum(new_species) / np.sum(new_species_prev))
        returned_probs.append(np.sum(returned_species) / np.sum(returned_species_prev))

    new_probs = np.array(new_probs)
    returned_probs = np.array(returned_probs)

    return {
        "new_probs": new_probs,
        "returned_probs": returned_probs
    }

def run_cross_subject_probability_analysis(dataset: dict, B, seed=0) -> dict:

    timeseries_tensor = np.array([dataset["baseline"], dataset["abx"], *dataset["post_abx_cohorts"]])
    cp = ColonizationProbability(timeseries_tensor)
    probs, abundances = cp.calc_probs()

    def perm_test(probs, B, seed=0):

        x_list, y_list = [], []
        for _, (x, y) in probs.items():

            x = float(x)
            y = float(y)
            x_list.append(x)
            y_list.append(y)

        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)

        d = x - y
        d = d[d != 0.0]

        T_obs = float(d.sum())

        mags = np.abs(d)

        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(B, mags.size), replace=True)
        d_perm = signs * mags

        T_perm = d_perm.sum(axis=1)

        p_value = float((np.sum(T_perm >= T_obs) + 1) / (B + 1))

        return p_value

    p_val = perm_test(probs, B=B, seed=seed)

    return {
        "probs": probs,
        "abundances": abundances,
        "p_value": p_val
    }
