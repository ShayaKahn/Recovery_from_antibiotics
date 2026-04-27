from src.host_specific_recovery.utils.general_utils import subset
import numpy as np

def species_proportions(baseline_spm, abx_smp, post_matrix, tau, strict=False, weighted=False):
    """
    This function calculates the proportions of species that survived, new, returned, and emerged during antibiotics.
    :param baseline_spm: Numpy array representing the baseline state of shape (#taxa, ) or (#baseline samples, #taxa).
    :param abx_smp: Numpy array representing the antibiotics state of shape (#taxa, ).
    :param post_matrix: Numpy array representing the post antibiotics state of shape (#time points, #taxa).
    :param tau: time point index to consider th cutoff between early and late return (the count of the time point
                starts from 1).
    :return: list of proportions [survived_prop, new_prop, returned_early_prop, returned_early_prop, emerged_prop]
    """

    if baseline_spm.ndim == 1:
        base = baseline_spm
    else:
        base = baseline_spm.sum(axis=1)
    post = post_matrix[-1, :]

    survived = ((base != 0) & (post != 0) & (abx_smp != 0))
    emerged = ((base == 0) & (post != 0) & (abx_smp != 0))
    new = ((base == 0) & (post != 0) & (abx_smp == 0))
    returned_species = subset(post_matrix, base, abx_smp, strict, new=False)

    early = [returned_species[i] for i in range(tau)]
    late = [returned_species[i] for i in range(tau, len(returned_species))]
    returned_early = np.sum(np.vstack(early), axis=0).astype(bool)
    returned_late = np.sum(np.vstack(late), axis=0).astype(bool)

    n_new = new.sum()
    n_early = returned_early.sum()
    n_late = returned_late.sum()
    n_survived = survived.sum()
    n_emerged = emerged.sum()

    if strict:
        s = n_new + n_early + n_late + n_survived + n_emerged
    else:
        s = np.size(np.nonzero(post))

    if weighted:
        return [post[survived].sum(), post[emerged].sum(), post[returned_early].sum(), post[returned_late].sum(),
                post[new].sum()]
    else:
        return [n_survived / s, n_emerged / s, n_early / s, n_late / s, n_new / s]

def run_species_proportios_heatmap_analysis(dataset: dict, tau: int, strict: bool = False) -> dict:

    # Initialization
    proportions = []
    weighted_proportions = []

    # Apply classification
    for i, abx, base in enumerate(zip(dataset["abx"], dataset["baseline"])):

        post_matrix = np.vstack([p[i, :] for p in dataset["post_abx_cohorts"]])
        proportions.append(species_proportions(base, abx, post_matrix, tau, strict=strict, weighted=False))
        weighted_proportions.append(species_proportions(base, abx, post_matrix, tau, strict=strict, weighted=True))

    return {
        "proportions": np.array(proportions),
        "weighted_proportions": np.array(weighted_proportions)
    }
