import numpy as np
from methods.glv import Glv
import random
import networkx as nx
from scipy.stats import powerlaw

class HC:
    # This class is responsible for the simulation of the historical contingency model.
    # I will write some notations to make the variables names more clear:
    # A: represents the interaction matrix in the GLV model.
    # s: represents the logistic growth vector in the GLV model.
    # r: represents the growth rate vector in the GLV model.
    # Y_0: represents the initial conditions matrix in the GLV model, where rows are samples and columns are species.
    # Y_p: represents the perturbed state in the GLV model. Each sample has a specific number of present species.
    # y: represents the test sample in the perturbed state. I choose the first sample that satisfies the steady state
    #    condition.
    # epsilon: represents the value to insert to the non-survived species. This value should be small and represent
    #         the effective amount of exposure of the system to the new species.
    # eta: represents the threshold to remove low abundances. This value should be small and represent the minimal
    #     abundance that the species should have to be considered as present.
    # y_s: represents the post perturbed state for the test sample. The steady state after inserting the total pool.
    # Y_s: represents the post perturbed state for the other samples. The steady state after inserting the total pool.
    def __init__(self, num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                 max_step, epsilon, eta, min_growth, max_growth, symmetric=True, alpha=None, method='RK45',
                 multiprocess=True, switch_off=False, n_jobs=4):
        # Inputs:
        # num_samples: The number of samples.
        # pool_size: The total size of the population.
        # num_survived_min: The minimal number of survived species.
        # num_survived_max: The maximal number of survived species.
        # mean: For the interaction matrix generation, the mean of the normal distribution.
        # sigma: For the interaction matrix generation, The standard deviation of the normal distribution.
        # c: The Connectance.
        # delta: The stop condition for the steady state.
        # final_time: The final time of the integration.
        # max_step: The maximal allowed step size.
        # epsilon: The value to insert to the non-survived species.
        # eta: The threshold to remove low abundances.
        # max_growth: The maximum growth rate.
        # min_growth: The minimum growth rate.
        # symmetric: If True, the interaction matrix will be symmetric.
        # alpha: The power-law exponent for the interaction matrix strength.
        # method: The method to solve the GLV model.
        # multiprocess: If True, the class will use the multiprocessing module.
        # switch_off: If True, the effect of the perturbed state species on the new inserted species is switched off and
        #             also, the effect of the new species on the perturbed species is also switched off.
        # n_jobs: The number of jobs to run in parallel.

        # input validation
        (self.num_samples, self.pool_size, self.num_survived_min, self.num_survived_max, self.mean, self.sigma, self.c,
         self.delta, self.final_time,
         self.max_step, self.epsilon, self.eta, self.min_growth, self.max_growth, self.symmetric, self.alpha,
         self.method, self.multiprocess,
         self.switch_off, self.n_jobs) = HC._validate_inputs(num_samples, pool_size, num_survived_min, num_survived_max,
                                                             mean, sigma, c, delta, final_time, max_step, epsilon,
                                                             eta, min_growth, max_growth, symmetric, alpha,
                                                             method, multiprocess, switch_off, n_jobs)
        # create the interaction matrix
        self.A = self._create_full_interaction_matrix()
        # create the logistic growth vector
        self.s = self._set_logistic_growth()
        # create the growth rate vector
        self.r = self._set_growth_rate()
        # create the list that represent the number of survived species for each sample
        self.num_survived_list = self._create_num_survived_list()
        # create the initial conditions matrix
        self.Y_0 = self._set_initial_conditions()
        # apply the GLV model and get the Y_p and the indexes were the steady state condition is not satisfied
        self.Y_p, self.event_not_satisfied_ind = self._apply_GLV(self.Y_0, norm=True, int_mat=self.A,
                                                                 n_samples=self.num_samples, n_jobs=self.n_jobs)
        # remove the low abundances and normalize Y_p
        self._filter_norm_Y_p()
        # get the index of the test sample and verify the steady state condition satisfied for this sample
        self.test_idx = self._define_test_index()
        # insert the total pool to the test sample in the perturbed state
        self.y = self._normalize_cohort(self._insert_total_pool_test())
        # apply the GLV model to y (the test sample)
        y_s, self.event_not_satisfied_ind_post = self._generate_y_s()
        # check if the steady state condition is satisfied
        assert not self.event_not_satisfied_ind_post, "The steady state condition not satisfied!"
        # remove the low abundances and normalize y_s
        self.y_s = self._normalize_cohort(self._remove_low_abundances(y_s))
        # remove the samples that the steady state condition is not satisfied
        self.Y_p = np.delete(self.Y_p, self.event_not_satisfied_ind, axis=0)
        self._modify_num_survived_list()
        # normalize Y_s
        self.Y_s = self._normalize_cohort(self._insert_total_pool_others())
        # apply the GLV model to Y_s
        self.Y_s, self.event_not_satisfied_ind_Y_s = self._apply_GLV(self.Y_s, norm=True, int_mat=self.A,
                                                                n_samples=self.Y_s.shape[0], n_jobs=self.n_jobs)
        self.Y_s = np.delete(self.Y_s, self.event_not_satisfied_ind_Y_s, axis=0)
        # remove the low abundances
        self.Y_s = self._remove_low_abundances(self.Y_s)

    @staticmethod
    def _validate_inputs(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                         max_step, epsilon, eta, min_growth, max_growth, symmetric, alpha, method, multiprocess,
                         switch_off, n_jobs):
        if not (isinstance(num_samples, int) and isinstance(pool_size, int) and
                isinstance(num_survived_min, int) and isinstance(num_survived_max, int)):
            raise ValueError("num_samples, pool_size, num_survived_min, and num_survived_max must be integers.")
        if num_samples <= 0 or pool_size <= 10 or num_survived_min <= 0 or num_survived_max <= 0:
            raise ValueError(
                "num_samples,"
                " num_survived_min and num_survived_max must be greater than 0 and pool_size must be greater then 10")
        if num_survived_max > pool_size or num_survived_min > num_survived_max:
            raise ValueError("num_survived_max must be smaller then pool_size and num_survived_min must be smaller or"
                             "equal to num_survived_max.")
        if not (isinstance(mean, float) or isinstance(mean, int)):
            raise ValueError("mean must be integer or float.")
        if not (isinstance(sigma, float) or isinstance(sigma, int) or 0 < sigma < 20):
            raise ValueError("sigma must be integer or float and in the interval [0, 20].")
        if method not in ['RK45', 'BDF', 'RK23', 'Radau', 'LSODA', 'DOP853']:
            raise ValueError("method must be one of the following methods: RK45, BDF, RK23, Radau, LSODA, and DOP853")
        if not (isinstance(c, float) and 0 < c < 1):
            raise ValueError("c must be float in the interval [0, 1].")
        if not (isinstance(eta, float) or isinstance(eta, int)):
            raise ValueError("eta must be a number.")
        if not (isinstance(eta, float) and 0 < eta < 1):
            raise ValueError("eta must be a number between 0 and 1.")
        # Check if delta is a number between 0 and 1
        if not isinstance(delta, float):
            raise ValueError("delta must be a float.")
        if not (0 < delta < 1):
            raise ValueError("delta must be a number between 0 and 1.")
        # Check if final_time is a number greater than zero
        if not (isinstance(final_time, (int, float)) and final_time > 0):
            raise ValueError("final_time must be a number greater than zero.")
        # Check if max_step is a number greater than zero and smaller than final_time
        if not (isinstance(max_step, (int, float)) and 0 < max_step < final_time):
            raise ValueError("max_step must be a number greater than zero and smaller than final_time.")
        if not (isinstance(epsilon, float) and 0 < epsilon < 1):
            raise ValueError("epsilon must be a number between 0 and 1.")
        if not (isinstance(max_growth, (float, int)) and 0 < max_growth <= 1):
            raise ValueError("max_growth must be a number between 0 and 1.")
        if not (isinstance(min_growth, (float, int)) and 0 <= min_growth <= 1):
            raise ValueError("min_growth must be a number between 0 and 1.")
        if min_growth > max_growth:
            raise ValueError("min_growth must be smaller or equal to max_growth.")
        if not ((isinstance(alpha, (float, int)) and alpha > 0) or alpha is None):
            raise ValueError("alpha must be a number greater than 0.")
        if not (isinstance(multiprocess, bool)):
            raise ValueError("multiprocess must be of type bool.")
        if not isinstance(symmetric, bool):
            raise ValueError("symmetric must be of type bool.")
        if not isinstance(switch_off, bool):
            raise ValueError("switch_off must be of type bool.")
        if not (isinstance(n_jobs, int)):
            raise ValueError("n_jobs must be of type int.")
        return (num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                max_step, epsilon, eta, min_growth, max_growth, symmetric, alpha, method, multiprocess,
                switch_off, n_jobs)

    def _create_num_survived_list(self):
        # This function creates a list of the number of survived species for each sample.
        # Returns:
        # A list of the number of survived species for each sample.

        if self.num_survived_max > self.num_survived_min:
            return np.random.randint(self.num_survived_min, self.num_survived_max+1, self.num_samples)
        elif self.num_survived_max == self.num_survived_min:
            return (np.ones(self.num_samples) * self.num_survived_min).astype(int)

    def _set_growth_rate(self):
        # This function sets the growth rate for each specie.
        # Returns:
        # Numpy array of the growth rate for each specie.

        return np.random.uniform(self.min_growth, self.max_growth, self.pool_size)

    def _set_logistic_growth(self):
        # This function sets the logistic growth for each specie.
        # Returns:
        # Numpy array of the logistic growth for each specie.

        return np.ones(self.pool_size)

    def _set_initial_conditions(self):
        # This function sets the initial conditions for the GLV model.
        # Returns:
        # Numpy matrix that contains the initial conditions for the GLV model.

        # initialize the initial conditions
        Y_0 = np.zeros((self.num_samples, self.pool_size))
        # create the initial conditions matrix
        survived_matrix = [random.sample(range(0, self.pool_size),
                                         self.num_survived_list[i]) for i in range(self.num_samples)]
        for index, (y, s) in enumerate(zip(Y_0, survived_matrix)):
            y[s] = np.random.rand(1, self.num_survived_list[index])
        return Y_0

    def _set_interaction_matrix(self):
        # This function sets the interaction matrix.
        # Returns:
        # Numpy matrix that represent the interaction matrix.

        # create the interaction matrix
        interaction_matrix = np.zeros((self.pool_size, self.pool_size))
        # generate the random numbers using the normal distribution
        random_numbers = np.random.normal(self.mean, self.sigma, size=(self.pool_size, self.pool_size))
        # create the mask base on the Connectance value
        mask = np.random.rand(self.pool_size, self.pool_size) < self.c
        # set the values of the interaction matrix
        interaction_matrix[mask] = random_numbers[mask]
        # set the diagonal to zero
        np.fill_diagonal(interaction_matrix, 0)
        return -np.abs(interaction_matrix)

    def _set_symmetric_interaction_matrix(self):
        # Returns:
        # Numpy matrix that represent the symmetric interaction matrix.

        # create the random graph
        G = nx.binomial_graph(self.pool_size, self.c)
        # create the mask
        mask = nx.to_numpy_array(G).astype(bool)
        # generate the random numbers using the normal distribution
        random_numbers = np.random.normal(self.mean, self.sigma, size=(self.pool_size, self.pool_size))
        # initialize the interaction matrix
        interaction_matrix = np.zeros((self.pool_size, self.pool_size))
        # set the values of the interaction matrix
        interaction_matrix[mask] = random_numbers[mask]
        # set the diagonal to zero
        np.fill_diagonal(interaction_matrix, 0)
        return -np.abs(interaction_matrix)

    def _create_full_interaction_matrix(self):
        # Returns:
        # Numpy matrix that represent the full interaction matrix.

        if self.symmetric:
            N = self._set_symmetric_interaction_matrix()
        else:
            N = self._set_interaction_matrix()
        # create the strength matrix
        H = self._set_int_strength_matrix()
        # create the full interaction matrix
        A = np.dot(N, H)
        return A

    def _generate_y_s(self):
        # Returns:
        # y_s: Numpy matrix that represent the post perturbed state.
        # event_not_satisfied_ind_y_s: The indices of the samples that the steady state condition is not satisfied.

        if self.switch_off:
            # switch off the effect of the perturbed species on the new inserted species and vice versa
            phi = self.Y_p[self.test_idx, :] != 0
            A_copy = self.A.copy()
            A_switch = self._switch_off_interactions(A_copy, phi)
            # apply the GLV model
            y_s, event_not_satisfied_ind_y_s = self._apply_GLV(self.y[None, :], norm=True, int_mat=A_switch,
                                                               n_samples=1, n_jobs=None)
        else:
            # apply the GLV model
            y_s, event_not_satisfied_ind_y_s = self._apply_GLV(self.y[None, :], norm=True, int_mat=self.A,
                                                               n_samples=1, n_jobs=None)
        return y_s, event_not_satisfied_ind_y_s

    def _set_int_strength_matrix(self):
        # Returns:
        # Numpy matrix that represent the interaction strength matrix.

        if self.alpha is None:
            # The case of equal interaction strength
            return np.diag(np.ones(self.pool_size))
        else:
            # The case of power-law distribution of the interaction strength
            diag = powerlaw.rvs(self.alpha, size=self.pool_size)
            mean = np.mean(diag)
            normalized_diag = diag / mean
            return np.diag(normalized_diag)

    @staticmethod
    def _switch_off_interactions(A, phi):
        # This function switches off the effect of the species in the set phi on the new inserted species and vice versa.
        # Inputs:
        # A: numpy matrix of shape (# species, # species) that represents the interaction matrix.
        # phi: the indexes of the perturbed species.
        # Returns:
        # A_copy: numpy matrix of shape (# species, # species) that represents the interaction matrix with the effect
        #         switched off.

        A_copy = A.copy()
        phi_c = np.setdiff1d(np.arange(A.shape[0]), phi)
        A_copy[np.ix_(phi_c, phi)] = 0
        A_copy[np.ix_(phi, phi_c)] = 0
        return A_copy

    def _apply_GLV(self, init_cond, norm, int_mat, n_samples, n_jobs):
        # Inputs:
        # init_cond: The initial conditions for the GLV model.
        # norm: If True, the function will normalize the output.
        # int_mat: the interaction matrix.
        # n_samples: The number of samples.
        # Returns:
        # final_abundances: Numpy matrix that represents the final abundances.

        glv_object = Glv(n_samples, self.pool_size, self.delta, self.r, self.s, int_mat, init_cond,
                         self.final_time, self.max_step, normalize=norm, method=self.method,
                         multiprocess=self.multiprocess, n_jobs=n_jobs)
        final_abundances = glv_object.solve()
        return final_abundances

    def _filter_norm_Y_p(self):
        #This function removes the low abundances and normalizes Y_p.

        # remove the low abundances
        self.Y_p[self.Y_p < self.eta] = 0
        # normalize the perturbed state
        self.Y_p = self._normalize_cohort(self.Y_p)

    def _define_test_index(self):
        # Returns:
        # test_idx: The index of the test sample.

        # get the indexes of the samples that the steady state condition is satisfied
        event_satisfied = np.setdiff1d(np.arange(0, self.num_samples), self.event_not_satisfied_ind)
        # get the index of the test sample and verify the steady state condition satisfied for this sample
        test_idx = event_satisfied[0]
        return test_idx

    def _modify_num_survived_list(self):
        # This function modifies the number of survived species list, by removing the samples that the steady state
        # condition is not satisfied.

        self.num_survived_list = [item for idx, item in enumerate(self.num_survived_list) if idx not in
                                  np.hstack([self.event_not_satisfied_ind, self.test_idx])]

    def _insert_total_pool_test(self):
        # This function inserts the total pool to y.
        # Returns:
        # y: Numpy matrix that represents y, the test sample with the total pool inserted.

        y = self.Y_p[self.test_idx, :].copy()
        mask = np.ones(self.pool_size, dtype=bool)
        idx = np.where(y != 0)
        mask[idx] = False
        # insert the total pool with amount epsilon
        y[mask] = self.epsilon
        return y

    def _insert_total_pool_others(self):
        # This function inserts the total pool to the other samples (not the tested sample).
        # Returns:
        # Y: Numpy matrix that represents the new perturbed state for the other samples.

        Y = self.Y_p.copy()
        Y = np.delete(Y, self.test_idx, axis=0)
        for p in Y:
            s = np.where(p != 0)
            mask = np.ones(p.shape[0], dtype=bool)
            mask[s] = False
            # insert the total pool with amount epsilon
            p[mask] = self.epsilon
        return Y

    def _remove_low_abundances(self, post):
        # Inputs:
        # post: Numpy matrix that represents the post perturbed state.
        # Returns:
        # post_copy: Numpy matrix that represents the post perturbed state after removing the low abundances.

        post_copy = post.copy()
        zero_ind = np.where(post_copy < self.eta)
        post_copy[zero_ind] = 0.0
        return self._normalize_cohort(post_copy)

    @staticmethod
    def _normalize_cohort(cohort):
        # normalization function
        if cohort.ndim == 1:
            cohort_normalized = cohort / cohort.sum()
        else:
            cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
        return cohort_normalized

    def get_results(self):
        #Returns:
        #results: Dictionary that contains the results.

        results = {
            "Y_p": self.Y_p,
            "y_s": self.y_s.squeeze(),
            "Y_s": self.Y_s,
        }
        return results
