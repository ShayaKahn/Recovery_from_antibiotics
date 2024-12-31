import numpy as np
cimport numpy as np
cimport numpy as cnp

def f(double t,  cnp.ndarray[cnp.float64_t, ndim=1] x, cnp.ndarray[cnp.float64_t, ndim=1] r,
      cnp.ndarray[cnp.float64_t, ndim=1] s, cnp.ndarray[cnp.float64_t, ndim=2] A, double delta):
    # RHS of the ODE
    return r * x - s * x ** 2 + x * np.dot(A, x)

def event(double t, cnp.ndarray[cnp.float64_t, ndim=1] x, cnp.ndarray[cnp.float64_t, ndim=1] r,
          cnp.ndarray[cnp.float64_t, ndim=1] s, cnp.ndarray[cnp.float64_t, ndim=2] A, double delta):
    # Event function for the ODE solver
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fx = f(t, x, r, s, A, delta)
    cdef double max_fx = np.max(np.abs(fx))
    return max_fx - delta
