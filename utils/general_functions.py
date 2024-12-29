import operator
import numpy as np
#from skbio.diversity import beta_diversity
from methods.similarity import Similarity
import random

def subset(post_matrix, base_sample, ABX_sample, strict, new):
    num_timepoints = post_matrix.shape[0]
    op_lst = [operator.ne] * num_timepoints
    if new:
        general_cond = [base_sample == 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    else:
        general_cond = [base_sample != 0, ABX_sample == 0, post_matrix[-1, :] != 0]
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


#def unifrac(base, abx, otu_ids, tree):
#    data = np.vstack([base, abx])
#   sample_ids = np.arange(0, data.shape[0], 1).tolist()
#    return beta_diversity(metric='unweighted_unifrac', counts=data,
#                           ids=sample_ids, taxa=otu_ids, tree=tree, validate=True)[1, 0]

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

def filter_data(df):
    # filter the data
    columns_to_check = df.columns[1:]
    mean_values = df[columns_to_check].mean(axis=1)
    condition_mask = mean_values >= 0.0001
    df = df[condition_mask]
    return df

def calc_similarity(steady_state, ABX, steady_state_mat, iters=1000):
    survived = np.where((steady_state != 0) & (ABX[0, :] != 0))[0]
    new = np.where((steady_state != 0) & (ABX[0, :] == 0))[0]
    synthetic_samples = [create_synthetic_cohort(steady_state_mat) for _ in range(iters)]
    sims_new = []
    sims_survived = []
    for j, abx in enumerate(steady_state_mat):
        print(j)
        sim_new = Similarity(abx[new], steady_state[new], method='Jaccard').calculate_similarity()
        sim_new_shuffled = np.array([Similarity(mat[j, :][new], steady_state[new], method='Jaccard').calculate_similarity()
                                     for mat in synthetic_samples])
        sim_new_z = z_score(sim_new, sim_new_shuffled)
        sim_survived = Similarity(abx[survived], steady_state[survived], method='Jaccard').calculate_similarity()
        sim_survived_shuffled = np.array([Similarity(mat[j, :][survived], steady_state[survived], method='Jaccard').calculate_similarity()
                                          for mat in synthetic_samples])
        sim_survived_z = z_score(sim_survived, sim_survived_shuffled)
        sims_new.append(sim_new_z)
        sims_survived.append(sim_survived_z)
    return sims_new, sims_survived

def create_synthetic_cohort(cohort):
    # initialize the synthetic cohort.
    synthetic_cohort = np.zeros(cohort.shape)
    # find the pool of species.
    pool = cohort[cohort != 0]
    # find the indices of the species.
    pool_indices = np.where(cohort != 0)[1]
    # iterate over the samples.
    for smp, real in zip(synthetic_cohort, cohort):
        pool_copy = pool.copy()
        pool_indices_copy = pool_indices.copy()
        # find the number of nonzero values in the sample.
        stop = np.size(np.nonzero(real))
        for _ in range(stop):
            # choose a random index.
            index = random.randint(0, len(pool_indices_copy) - 1)
            # update the synthetic sample.
            smp[pool_indices_copy[index]] = pool_copy[index]
            # update the pool.
            mask = pool_indices_copy != pool_indices_copy[index]
            pool_copy = pool_copy[mask]
            pool_indices_copy = pool_indices_copy[mask]
    return synthetic_cohort
