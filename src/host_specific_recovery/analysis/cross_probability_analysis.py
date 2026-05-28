from src.host_specific_recovery.utils.general_utils import subset
from src.host_specific_recovery.statistical_models.colonization_probability import ColonizationProbability
import numpy as np

def run_cross_species_probability_analysis(dataset: dict, min_estimate) -> dict:

    post_ABX = dataset["post_abx_cohorts"]
    baseline = dataset["baseline"]
    ABX = dataset["abx"]

    new_probs = []
    returned_probs = []

    for i, (base, abx) in enumerate(zip(baseline, ABX)):

        post_matrix = np.vstack([p[i, :] for p in post_ABX])
        new_species = subset(post_matrix, base, abx, True, True)[-2]
        returned_species = subset(post_matrix, base, abx, False, False)[-2]

        dim = post_matrix.shape[0] - 1
        new_species_prev = subset(post_matrix[:dim, :], base, abx, True, True)[-1]
        returned_species_prev = subset(post_matrix[:dim, :], base, abx, False, False)[-1]

        if not ((np.sum(new_species_prev) <= min_estimate) or (np.sum(returned_species_prev) <= min_estimate)):

            print(np.sum(new_species_prev))
            print(np.sum(new_species))
            print(" ")
            new_probs.append(np.sum(new_species) / np.sum(new_species_prev))
            returned_probs.append(np.sum(returned_species) / np.sum(returned_species_prev))

    new_probs = np.array(new_probs)
    returned_probs = np.array(returned_probs)

    return {
        "new_probs": new_probs,
        "returned_probs": returned_probs
    }

def run_cross_subject_probability_analysis(dataset: dict, B=10000, seed=0) -> dict:

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


def run_cross_subject_probability_analysis_abundance(outputs_cross_subject: dict) -> dict:
    """Collect complete-pair and pooled abundance values by taxon type.

    Each value in ``outputs_cross_subject["abundances"]`` is expected to contain
    returned-taxa abundances at index 0 and new-taxa abundances at index 1.
    The paired matrices keep only taxa with both abundance values present, while
    the pooled arrays keep all finite values.
    """
    abundances = outputs_cross_subject.get("abundances")
    if abundances is None:
        raise KeyError("outputs_cross_subject must contain an 'abundances' entry")

    def collect_abundances(group_index):
        complete_pairs = []
        finite_values = []

        for abundance_pair in abundances.values():
            values = np.asarray(abundance_pair[group_index], dtype=float)
            finite_mask = np.isfinite(values)

            if np.all(finite_mask):
                complete_pairs.append(values)

            finite_values.extend(values[finite_mask])

        if complete_pairs:
            complete_pairs = np.vstack(complete_pairs)
        else:
            complete_pairs = np.empty((0, 2), dtype=float)

        return complete_pairs, np.asarray(finite_values, dtype=float)

    ret_taxa_abundances_mat, ret_taxa_abundances_all_mat = collect_abundances(group_index=0)
    new_taxa_abundances_mat, new_taxa_abundances_all_mat = collect_abundances(group_index=1)

    return {
        "new_taxa_abundances": new_taxa_abundances_mat,
        "new_taxa_abundances_all_mat": new_taxa_abundances_all_mat,
        "ret_taxa_abundances": ret_taxa_abundances_mat,
        "ret_taxa_abundances_all_mat": ret_taxa_abundances_all_mat
    }
