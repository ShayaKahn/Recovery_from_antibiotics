import numpy as np
cimport numpy as np
cimport numpy as cnp

def f(double t, double[:] x, double[:] r, double[:] s, double[:, :] A, double delta):
    # RHS of the ODE
    cdef int i, p
    cdef int n = np.size(x)
    return np.array([r[i] * x[i] - s[i] * x[i] ** 2 + sum([A[i, p] * x[
           i] * x[p] for p in range(n) if p != i]) for i in range(n)])

def g(double t,  cnp.ndarray[cnp.float64_t, ndim=1] x, cnp.ndarray[cnp.float64_t, ndim=1] r,
      cnp.ndarray[cnp.float64_t, ndim=1] s, cnp.ndarray[cnp.float64_t, ndim=2] A, double delta):
    # RHS of the ODE
    return r * x - s * x ** 2 + x * np.dot(A, x)

def event(double t, double[:] x, double[:] r, double[:] s, double[:, :] A, double delta):
    # Event function for the ODE solver
    cdef double[:] fx = f(t, x, r, s, A, delta)
    cdef double max_fx = np.max(np.abs(fx))
    return max_fx - delta