import operator
import numpy as np
from skbio.diversity import beta_diversity

def subset(post_matrix, base_sample, ABX_sample, strict, new):
    num_timepoints = post_matrix.shape[0]
    op_lst = [operator.ne] * num_timepoints
    if new:
        general_cond = [base_sample == 0, ABX_sample == 0]
    else:
        general_cond = [base_sample != 0, ABX_sample == 0]
    timepoints_vals = []
    if strict:
        for j in range(num_timepoints):
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            special_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1, num_timepoints)]
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                op_lst[j] = operator.eq
    else:
        # iterate over the time points.
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            # combine the conditions.
            cond = general_cond + inter_cond
            # find the returned species at each time point.
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
                op_lst[j] = operator.eq
    return timepoints_vals


def create_cohort_dict(keys, special_key, cohort):
    cohort_dict = {}
    for i, key in enumerate(keys):
        if key != special_key:
            if type(cohort) is not list:
                cohort_dict[key] = cohort[i, :].reshape(1, -1)
            else:
                cohort_dict[key] = cohort[i]
    return cohort_dict


def z_score(p, smp):
    mu = np.mean(np.hstack(smp))
    sigma = np.std(smp)
    return (p - mu) / sigma


def create_shuffled_cohort(cohort):
    shuffled_cohort = np.zeros(cohort.shape)

    pool = cohort[cohort != 0]
    pool_indices = np.where(cohort != 0)[1]

    for smp, real in zip(shuffled_cohort, cohort):
        pool_copy = pool.copy()
        pool_indices_copy = pool_indices.copy()
        stop = np.size(np.nonzero(real))
        for _ in range(stop):
            index = random.randint(0, len(pool_indices_copy) - 1)
            smp[pool_indices_copy[index]] = pool_copy[index]
            mask = pool_indices_copy != pool_indices_copy[index]
            pool_copy = pool_copy[mask]
            pool_indices_copy = pool_indices_copy[mask]
    return normalize_cohort(shuffled_cohort)


def unifrac(base, abx, otu_ids, tree):
    data = np.vstack([base, abx])
    sample_ids = np.arange(0, data.shape[0], 1).tolist()
    return beta_diversity(metric='unweighted_unifrac', counts=data,
                           ids=sample_ids, taxa=otu_ids, tree=tree, validate=True)[1, 0]