import operator
import numpy as np
from skbio.diversity import beta_diversity
from methods.similarity import Similarity
import random

def subset(post_matrix, base_sample, ABX_sample, strict, new):

    # This function find the return times of the new or the returned species.
    # Inputs:
    # post_matrix: numpy matrix of shape (# samples, # species) that contains the post
    #              antibiotics samples of the subject ordered chronologically in the rows.
    # base_sample: numpy array of shape (# species,) that contains the baseline sample of the subject.
    # ABX_sample: numpy array of shape (# species,) that contains the antibiotics sample of the subject.
    # strict: boolean that indicates if the condition is strict or not.
    # new: boolean that indicates if the function is looking for the new species or the returned species.
    # Returns:
    # timepoints_vals: list of boolean numpy arrays of shape (# species,) that represent the new or returned species at
    # each time point.

    # initialize the operators list.
    num_timepoints = post_matrix.shape[0]
    op_lst = [operator.ne] * num_timepoints
    if new:
        # define the general condition for the new species.
        general_cond = [base_sample == 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    else:
        # define the general condition for the returned species.
        general_cond = [base_sample != 0, ABX_sample == 0, post_matrix[-1, :] != 0]
    timepoints_vals = []
    if strict:
        # iterate over the time points.
        for j in range(num_timepoints):
            # define the intermediate condition.
            inter_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1)]
            # define the special condition.
            special_cond = [op_lst[i](post_matrix[i, :], 0) for i in range(j + 1, num_timepoints)]
            # combine the conditions.
            cond = general_cond + inter_cond + special_cond
            timepoints_vals.append(np.logical_and.reduce(cond))
            if j != num_timepoints - 1:
                # update the operators list.
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
    # This function creates a dictionary that maps subjects to their corresponding samples,
    # excluding the specific subject.
    # Inputs:
    # keys: list of strings that represent the identifiers of the subjects.
    # special_key: string that represents the identifier of the subject to exclude.
    # cohort: numpy matrix of shape (# subjects, # species) or list that contains numpy arrays of shape (# species,)
    #         that contains the samples of the subjects.
    # Returns:
    # cohort_dict: dictionary that maps subjects to their corresponding samples, excluding the specific subject.

    assert len(keys) == cohort.shape[0], "The number of keys should be equal to the number of subjects."
    # initialize the cohort dictionary.
    cohort_dict = {}
    # iterate over the keys.
    for i, key in enumerate(keys):
        # exclude the specific subject.
        if key != special_key:
            if type(cohort) is not list:
                cohort_dict[key] = cohort[i, :].reshape(1, -1)
            else:
                cohort_dict[key] = cohort[i]
    return cohort_dict


def z_score(p, smp):
    # Inputs:
    # p: float that represents the specific value.
    # smp: numpy array that represent the sample of values.
    # Returns:
    # z-score of the specific value.

    mu = np.mean(smp)
    sigma = np.std(smp)
    return (p - mu) / sigma


def create_shuffled_cohort(cohort):
    # initialize the shuffled cohort.
    shuffled_cohort = np.zeros(cohort.shape)

    # find the pool of species.
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
    # This function calculates the unweighted unifrac distance between two samples.
    # Inputs:
    # base: numpy array of shape (# species,) that represents the baseline sample.
    # abx: numpy array of shape (# species,) that represents the antibiotics sample.
    # otu_ids: list of strings that represent the identifiers of the species.
    # tree: skbio.tree.TreeNode that represents the phylogenetic tree of the species.
    # Returns:
    # float that represents the unweighted unifrac distance between the two samples.

    data = np.vstack([base, abx])
    sample_ids = np.arange(0, data.shape[0], 1).tolist()
    return beta_diversity(metric='unweighted_unifrac', counts=data,
                           ids=sample_ids, taxa=otu_ids, tree=tree, validate=True)[1, 0]

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

def filter_data(df):
    # This function filters the data by removing the samples with low mean values.
    # Inputs:
    # df: pandas DataFrame that contains the cohort, the columns represent the samples.
    # Returns:
    # pandas DataFrame that contains the filtered cohort.

    # filter the data
    columns_to_check = df.columns[1:]
    mean_values = df[columns_to_check].mean(axis=1)
    condition_mask = mean_values >= 0.0001
    df = df[condition_mask]
    return df

def calc_similarity(steady_state, ABX, steady_state_mat, iters=1000):
    # This function calculates the similarity values between the steady state of particular subject and the steady states
    # of other subjects for the subset of the new species and the subset of the survived species. The similarity values
    # are standarized using the z-score calculated with respect to synthetic samples.
    # Inputs:
    # steady_state: numpy array of shape (# species,) that represents the steady state of the particular subject.
    # ABX: numpy array of shape (# species,) that represents the antibiotics sample of the particular subject.
    # steady_state_mat: numpy matrix of shape (# subjects, # species) that contains the steady states of the subjects.
    # iters: int that represents the number of synthetic samples to generate.
    # Returns:
    # sims_new: list of similarity values between the new species of the particular subject and the new species of other
    #           subjects.
    # sims_survived: list of similarity values between the survived species of the particular subject and the survived
    #                species of other subjects.

    # find the indices of the new and the survived species.
    survived = np.where((steady_state != 0) & (ABX[0, :] != 0))[0]
    # find the indices of the new species.
    new = np.where((steady_state != 0) & (ABX[0, :] == 0))[0]
    # generate synthetic samples.
    synthetic_samples = [create_synthetic_cohort(steady_state_mat) for _ in range(iters)]
    # initialize the lists of similarity values.
    sims_new = []
    sims_survived = []
    # iterate over the steady states.
    for j, s in enumerate(steady_state_mat):
        sim_new = Similarity(s[new], steady_state[new], method='Jaccard').calculate_similarity()
        sim_new_shuffled = np.array([Similarity(mat[j, :][new], steady_state[new],
                                                method='Jaccard').calculate_similarity()
                                     for mat in synthetic_samples])
        sim_new_z = z_score(sim_new, sim_new_shuffled)
        sim_survived = Similarity(s[survived], steady_state[survived], method='Jaccard').calculate_similarity()
        sim_survived_shuffled = np.array([Similarity(mat[j, :][survived], steady_state[survived],
                                                     method='Jaccard').calculate_similarity()
                                          for mat in synthetic_samples])
        sim_survived_z = z_score(sim_survived, sim_survived_shuffled)
        sims_new.append(sim_new_z)
        sims_survived.append(sim_survived_z)
    return sims_new, sims_survived

def create_synthetic_cohort(cohort):
    # This function creates a synthetic cohort.
    # Inputs:
    # cohort: numpy matrix of shape (# subjects, # species) that contains the samples of the subjects.
    # Returns:
    # synthetic_cohort: numpy matrix of shape (# subjects, # species) that contains the synthetic samples of the subjects.

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
