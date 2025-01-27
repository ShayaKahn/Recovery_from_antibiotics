import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand

def _create_synthetic_cohort(cnp.ndarray[cnp.float64_t, ndim=2] cohort):
    """
    This method creates a synthetic cohort by generating samples with the same size (number of nonzero values in
    each sample) of the input reference cohort. The samples are generated baced on the distribution of their
    frequency.
    Input:
    cohort: numpy matrix of shape (# samples, # species) that represent the reference cohort.
    Returns: numpy matrix of shape (# samples, # species) that represent the synthetic cohort.
    """
    # initialize the synthetic cohort.
    cdef cnp.ndarray[cnp.float64_t, ndim=2] synthetic_cohort = np.zeros_like(cohort)
    cdef int stop
    # find the pool of species.
    cdef cnp.ndarray[cnp.int64_t, ndim = 1, cast=True] pool = cohort[cohort != 0]
    # find the indices of the species.
    cdef cnp.ndarray[cnp.int64_t, ndim = 1] pool_indices = np.where(cohort != 0)[1]
    cdef cnp.ndarray[cnp.int8_t, ndim = 1, cast=True] mask
    # iterate over the samples.
    for smp, real in zip(synthetic_cohort, cohort):
        pool_copy = pool.copy()
        pool_indices_copy = pool_indices.copy()
        # find the number of nonzero values in the sample.
        stop = np.size(np.nonzero(real))
        for _ in range(stop):
            # choose a random index.
            index = rand() % pool_indices_copy.shape[0]
            # update the synthetic sample.
            smp[pool_indices_copy[index]] = pool_copy[index]
            # update the pool.
            mask = pool_indices_copy != pool_indices_copy[index]
            pool_copy = pool_copy[mask]
            pool_indices_copy = pool_indices_copy[mask]
    return synthetic_cohort