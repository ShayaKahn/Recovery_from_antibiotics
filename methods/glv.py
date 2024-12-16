from scipy.integrate import solve_ivp
from cython_modules.glv_functions import f, g, event
from dask import delayed, compute
from dask.distributed import Client
import numpy as np

class Glv:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, r, s, interaction_matrix, initial_cond, final_time, max_step,
                 normalize=True, method='RK45', multiprocess=True):
        """
        Inputs:
        n_samples: The number of samples you are need to compute.
        n_species: The number of species at each sample.
        delta: This parameter is responsible for the stop condition at the steady state.
        r: growth rate vector of shape (n_species,).
        s: logistic growth term vector of size (n_species,).
        interaction_matrix: interaction matrix of shape (n_species, n_species).
        initial_cond: set of initial conditions for each sample, the shape is (n_samples, n_species)
        final_time: the final time of the integration.
        max_step: maximal allowed step size.
        normalize: boolean, if True the function normalize the output.
        method: method to solve the ODE, default is 'RK45'.
        multiprocess: boolean, if True the function will use dask to parallelize the computation.
        """

        (self.smp, self.n, self.delta, self.r, self.s, self.A, self.Y, self.final_time, self.max_step, self.normalize,
         self.method, self.multiprocess) = Glv._validate_input(n_samples, n_species, delta, r, s, interaction_matrix,
                                                               initial_cond, final_time, max_step, normalize, method,
                                                               multiprocess)

        if self.multiprocess:
            self.client = Client(n_workers=10, threads_per_worker=1)

        # Initialization
        self.Final_abundances = np.zeros((self.n, self.smp))
        self.Final_abundances_single_sample = np.zeros(self.n)

    @ staticmethod
    def _validate_input(n_samples, n_species, delta, r, s, interaction_matrix, initial_cond, final_time, max_step,
                        normalize, method, multiprocess):
        # Check if n_samples and n_species are integers greater than 0
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be an integer greater than 0.")
        if not isinstance(n_species, int) or n_species <= 0:
            raise ValueError("n_species must be an integer greater than 0.")

        # Check if delta is a number between 0 and 1
        if not (0 < delta < 1):
            raise ValueError("delta must be a number between 0 and 1.")

        # Check if r and s are numpy vectors of length n_species
        if not (isinstance(r, np.ndarray) and r.shape == (n_species,)):
            raise ValueError("r must be a numpy vector of length n_species.")
        if not (isinstance(s, np.ndarray) and s.shape == (n_species,)):
            raise ValueError("s must be a numpy vector of length n_species.")

        # Check if interaction_matrix is a numpy matrix of shape (n_species, n_species)
        if not (isinstance(interaction_matrix, np.ndarray) and interaction_matrix.shape == (n_species, n_species)):
            raise ValueError("interaction_matrix must be a numpy matrix of shape (n_species, n_species).")

        # Check if initial_cond is a numpy matrix of shape (n_species, n_samples) with non-negative values
        if not (isinstance(initial_cond, np.ndarray) and initial_cond.shape == (n_samples, n_species) and np.all(
                initial_cond >= 0)):
            raise ValueError(
                "initial_cond must be a numpy matrix of shape (n_samples, n_species) with non-negative values.")

        # Check if final_time is a number greater than zero
        if not (isinstance(final_time, (int, float)) and final_time > 0):
            raise ValueError("final_time must be a number greater than zero.")

        # Check if max_step is a number greater than zero and smaller than final_time
        if not (isinstance(max_step, (int, float)) and 0 < max_step < final_time):
            raise ValueError("max_step must be a number greater than zero and smaller than final_time.")

        if not (isinstance(multiprocess, bool)):
            raise ValueError("multiprocess must be of type bool.")

        return (n_samples, n_species, delta, r, s, interaction_matrix, initial_cond, final_time, max_step, normalize,
                method, multiprocess)

    def solve(self):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """
        # Set the parameters to the functions f and event.
        #f_with_params = lambda t, x: f(t, x, self.r, self.s, self.A, self.delta)
        f_with_params = lambda t, x: g(t, x, self.r, self.s, self.A, self.delta)
        event_with_params = lambda t, x: event(t, x, self.r, self.s, self.A, self.delta)

        # event definitions
        event_with_params.terminal = True
        event_with_params.direction = -1

        event_not_satisfied_ind = []

        if self.multiprocess:

            sol_objects = [self.solve_for_m(f_with_params, event_with_params, m) for m in range(self.smp)]
            solutions = compute(*sol_objects)

            for m, sol in enumerate(solutions):
                self.Final_abundances[:, m] = sol.y[:, -1]
                zero_ind = np.where(self.Final_abundances[:, m] < 0.0)
                self.Final_abundances[:, m][zero_ind] = 0.0

                if np.size(sol.t_events[0]) == 0:
                    event_not_satisfied_ind.append(m)
        else:
            for m in range(self.smp):
                print(m)
                # solve GLV up to time span.
                sol = solve_ivp(f_with_params, (0, self.final_time), self.Y[m, :], max_step=self.max_step,
                                events=event_with_params, method=self.method)

                self.Final_abundances[:, m] = sol.y[:, -1]
                zero_ind = np.where(self.Final_abundances[:, m] < 0.0)
                self.Final_abundances[:, m][zero_ind] = 0.0

                if np.size(sol.t_events[0]) == 0:
                    event_not_satisfied_ind.append(m)

        final_abundances = self.Final_abundances
        if self.normalize:
            return self.normalize_cohort(final_abundances.T), event_not_satisfied_ind
        else:
            return final_abundances.T, event_not_satisfied_ind

    @delayed
    def solve_for_m(self, f_with_params, event_with_params,  m):
        sol = solve_ivp(f_with_params, (0, self.final_time), self.Y[m, :], max_step=self.max_step,
                        events=event_with_params, method=self.method)
        return sol

    def close_client(self):
        if self.multiprocess:
            self.client.close()

    @staticmethod
    def normalize_cohort(cohort):
        # normalization function
        if cohort.ndim == 1:
            cohort_normalized = cohort / cohort.sum()
        else:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
        return cohort_normalized