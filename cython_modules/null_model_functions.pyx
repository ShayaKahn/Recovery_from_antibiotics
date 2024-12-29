import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, srand
from libc.time cimport time

def generate_samples(cnp.ndarray[cnp.float64_t, ndim=1] pool,
                     cnp.ndarray[cnp.int32_t, ndim=1] indices, int stop, int num_reals, int size):

    # Define variables with Cython types
    cdef int k = num_reals
    cdef int p = size
    cdef int n = stop
    cdef cnp.ndarray[cnp.float64_t, ndim=2] matrix = np.zeros((num_reals, p), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] row
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pool_copy
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices_copy
    cdef int i, index
    cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] mask

    # Seed the random number generator
    srand(time(NULL))

    for i in range(k):
        pool_copy = pool.copy()
        indices_copy = indices.copy()
        row = matrix[i, :]
        for _ in range(n):
            index = rand() % indices_copy.shape[0]
            row[indices_copy[index]] = pool_copy[index]
            mask = indices_copy != indices_copy[index]
            pool_copy = pool_copy[mask]
            indices_copy = indices_copy[mask]

    return matrix