import numpy as np
from methods.glv import Glv
import random
import networkx as nx
from scipy.stats import powerlaw

class HC:
    """
    This class is responsible for the simulation of the historical contingency model.
    """
    def __init__(self, num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                 max_step, epsilon, threshold, min_growth, max_growth, symmetric=True, alpha=None, method='RK45',
                 multiprocess=True, switch_off=False):
        """
        Inputs:
        num_samples: The number of samples.
        pool_size: The total size of the population.
        num_survived_min: The minimal number of survived species.
        num_survived_max: The maximal number of survived species.
        mean: For the interaction matrix generation, the mean of the normal distribution.
        sigma: For the interaction matrix generation, The standard deviation of the normal distribution.
        c: The Connectance.
        delta: The stop condition for the steady state.
        final_time: The final time of the integration.
        max_step: The maximal allowed step size.
        epsilon: The value to insert to the non-survived species.
        threshold: The threshold to remove low abundances.
        max_growth: The maximum growth rate.
        min_growth: The minimum growth rate.
        symmetric: If True, the interaction matrix will be symmetric.
        alpha: The power-law exponent for the interaction matrix strength.
        method: The method to solve the GLV model.
        multiprocess: If True, the class will use the multiprocessing module.
        switch_off: If True, the effect of the perturbed state species on the new inserted species is switched off and
                    also, the effect of the new species on the perturbed species is also switched off.
        """
        (self.num_samples, self.pool_size, self.num_survived_min, self.num_survived_max, self.mean, self.sigma, self.c,
         self.delta, self.final_time,
         self.max_step, self.epsilon, self.threshold, self.min_growth, self.max_growth, self.symmetric, self.alpha,
         self.method, self.multiprocess,
         self.switch_off) = HC._validate_inputs(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma,
                                                c, delta, final_time,
                                                max_step, epsilon, threshold, min_growth, max_growth, symmetric, alpha,
                                                method, multiprocess, switch_off)
        self.num_survived_list = self._create_num_survived_list()
        self.r = self._set_growth_rate()
        self.s = self._set_logistic_growth()
        self.init_cond = self._set_initial_conditions()
        self.A = self._create_full_interaction_matrix()
        self.perturbed_state, self.event_not_satisfied_ind = self._apply_GLV(self.init_cond, norm=True, int_mat=self.A,
                                                                             n_samples=self.num_samples)
        self.perturbed_state[self.perturbed_state < self.threshold] = 0
        self.perturbed_state = self._normalize_cohort(self.perturbed_state)
        self.event_satisfied = np.setdiff1d(np.arange(0, num_samples), self.event_not_satisfied_ind)
        self.test_idx = self.event_satisfied[0]
        self.new_perturbed_state = self._normalize_cohort(self._insert_total_pool())
        self.post_perturbed_state, self.event_not_satisfied_ind_post = self._generate_post_perturbed_state()
        assert not self.event_not_satisfied_ind_post, "The steady state condition not satisfied!"
        self.filtered_post_perturbed_state = self._remove_low_abundances(self.post_perturbed_state)
        self.filtered_post_perturbed_state = self._normalize_cohort(self.filtered_post_perturbed_state)
        self.perturbed_state = np.delete(self.perturbed_state, self.event_not_satisfied_ind, axis=0)
        self.num_survived_list = [item for idx, item in enumerate(self.num_survived_list) if idx not in
                                  np.hstack([self.event_not_satisfied_ind, self.test_idx])]
        self.new_perturbed_state_others = self._normalize_cohort(self._insert_total_pool_others())
        self.post_perturbed_state_others, event_not_satisfied_ind_post_others = self._apply_GLV(
            self.new_perturbed_state_others,
            norm=True, int_mat=self.A,
            n_samples=self.new_perturbed_state_others.shape[0])
        self.post_perturbed_state_others = np.delete(self.post_perturbed_state_others,
                                                     event_not_satisfied_ind_post_others, axis=0)
        self.filtered_post_perturbed_state_others = self._remove_low_abundances(self.post_perturbed_state_others)

    @staticmethod
    def _validate_inputs(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                         max_step, epsilon, threshold, min_growth, max_growth, symmetric, alpha, method, multiprocess,
                         switch_off):
        if not (isinstance(num_samples, int) and isinstance(pool_size, int) and
                isinstance(num_survived_min, int) and isinstance(num_survived_max, int)):
            raise ValueError("num_samples, pool_size, num_survived_min, and num_survived_max must be integers.")
        if num_samples <= 0 or pool_size <= 10 or num_survived_min <= 0 or num_survived_max <= 0:
            raise ValueError(
                "num_samples,"
                " num_survived_min and num_survived_max must be greater than 0 and pool_size must be greater then 10")
        if num_survived_max > pool_size or num_survived_min > num_survived_max:
            raise ValueError("num_survived_max must be smaller then pool_size and num_survived_min must be smaller then"
                             " num_survived_max.")
        if not (isinstance(mean, float) or isinstance(mean, int)):
            raise ValueError("mean must be integer or float.")
        if not (isinstance(sigma, float) or isinstance(sigma, int) or 0 < sigma < 20):
            raise ValueError("sigma must be integer or float and in the interval [0, 20].")
        if method not in ['RK45', 'BDF', 'RK23', 'Radau', 'LSODA', 'DOP853']:
            raise ValueError("method must be one of the following methods: RK45, BDF, RK23, Radau, LSODA, and DOP853")
        if not (isinstance(c, float) and 0 < c < 1):
            raise ValueError("c must be float in the interval [0, 1].")
        if not (isinstance(threshold, float) or isinstance(threshold, int)):
            raise ValueError("threshold must be a number.")
        if not (isinstance(threshold, float) and 0 < threshold < 1):
            raise ValueError("threshold must be a number between 0 and 1.")
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
        if not (isinstance(min_growth, (float, int)) and 0 <= min_growth < 1):
            raise ValueError("min_growth must be a number between 0 and 1.")
        if min_growth > max_growth:
            raise ValueError("min_growth must be smaller than max_growth.")
        if not ((isinstance(alpha, (float, int)) and alpha > 0) or alpha is None):
            raise ValueError("alpha must be a number greater than 0.")
        if not (isinstance(multiprocess, bool)):
            raise ValueError("multiprocess must be of type bool.")
        if not isinstance(symmetric, bool):
            raise ValueError("symmetric must be of type bool.")
        if not isinstance(switch_off, bool):
            raise ValueError("switch_off must be of type bool.")
        return (num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta, final_time,
                max_step, epsilon, threshold, min_growth, max_growth, symmetric, alpha, method, multiprocess,
                switch_off)

    def _create_num_survived_list(self):
        return np.random.randint(self.num_survived_min, self.num_survived_max, self.num_samples)

    def _set_growth_rate(self):
        return np.random.uniform(self.min_growth, self.max_growth, self.pool_size)

    def _set_logistic_growth(self):
        return np.ones(self.pool_size)

    def _set_initial_conditions(self):
        y0 = np.zeros((self.num_samples, self.pool_size))
        survived_matrix = [random.sample(range(0, self.pool_size),
                                         self.num_survived_list[i]) for i in range(self.num_samples)]
        for index, (y, s) in enumerate(zip(y0, survived_matrix)):
            y[s] = np.random.rand(1, self.num_survived_list[index])
        return y0

    def _set_interaction_matrix(self):
        interaction_matrix = np.zeros((self.pool_size, self.pool_size))
        random_numbers = np.random.normal(self.mean, self.sigma, size=(self.pool_size, self.pool_size))
        mask = np.random.rand(self.pool_size, self.pool_size) < self.c
        interaction_matrix[mask] = random_numbers[mask]
        np.fill_diagonal(interaction_matrix, 0)
        return -np.abs(interaction_matrix)

    def _set_symmetric_interaction_matrix(self):
        G = nx.binomial_graph(self.pool_size, self.c)
        mask = nx.to_numpy_array(G).astype(bool)
        random_numbers = np.random.normal(self.mean, self.sigma, size=(self.pool_size, self.pool_size))
        interaction_matrix = np.zeros((self.pool_size, self.pool_size))
        interaction_matrix[mask] = random_numbers[mask]
        np.fill_diagonal(interaction_matrix, 0)
        return -np.abs(interaction_matrix)

    def _create_full_interaction_matrix(self):
        if self.symmetric:
            N = self._set_symmetric_interaction_matrix()
        else:
            N = self._set_interaction_matrix()
        H = self._set_int_strength_matrix()
        A = np.dot(N, H)
        return A

    def _generate_post_perturbed_state(self):
        if self.switch_off:
            perturbed_species = self.perturbed_state[self.test_idx, :] != 0
            A_copy = self.A.copy()
            A_switch = self._switch_off_interactions(A_copy, perturbed_species)
            post_perturbed_state, event_not_satisfied_ind_post = self._apply_GLV(self.new_perturbed_state[None, :],
                                                                                 norm=True,
                                                                                 int_mat=A_switch,
                                                                                 n_samples=1)
        else:
            post_perturbed_state, event_not_satisfied_ind_post = self._apply_GLV(self.new_perturbed_state[None, :],
                                                                                 norm=True, int_mat=self.A,
                                                                                 n_samples=1)
        return post_perturbed_state, event_not_satisfied_ind_post

    def _set_int_strength_matrix(self):
        if self.alpha is None:
            return np.diag(np.ones(self.pool_size))
        else:
            diag = powerlaw.rvs(self.alpha, size=self.pool_size)
            mean = np.mean(diag)
            normalized_diag = diag / mean
            return np.diag(normalized_diag)

    @staticmethod
    def _switch_off_interactions(A, perturbed_species):
        A_copy = A.copy()
        perturbed_species_c = np.setdiff1d(np.arange(A.shape[0]), perturbed_species)
        A_copy[np.ix_(perturbed_species_c, perturbed_species)] = 0
        A_copy[np.ix_(perturbed_species, perturbed_species_c)] = 0
        return A_copy

    def _apply_GLV(self, init_cond, norm, int_mat, n_samples):
        glv_object = Glv(n_samples, self.pool_size, self.delta, self.r, self.s, int_mat, init_cond,
                         self.final_time, self.max_step, normalize=norm, method=self.method,
                         multiprocess=self.multiprocess)
        final_abundances = glv_object.solve()
        return final_abundances

    def _insert_total_pool(self):
        new_perturbed_state = self.perturbed_state[self.test_idx, :].copy()
        mask = np.ones(self.pool_size, dtype=bool)
        idx = np.where(new_perturbed_state != 0)
        mask[idx] = False
        new_perturbed_state[mask] = self.epsilon
        return new_perturbed_state

    def _insert_total_pool_others(self):
        new_perturbed_state = self.perturbed_state.copy()
        new_perturbed_state = np.delete(new_perturbed_state, self.test_idx, axis=0)
        for p in new_perturbed_state:
            s = np.where(p != 0)
            mask = np.ones(p.shape[0], dtype=bool)
            mask[s] = False
            p[mask] = self.epsilon
        return new_perturbed_state

    def _remove_low_abundances(self, post):
        post_copy = post.copy()
        zero_ind = np.where(post_copy < self.threshold)
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
        results = {
            "perturbed_state": self.perturbed_state,
            "filtered_post_perturbed_state": self.filtered_post_perturbed_state.squeeze(),
            "filtered_post_perturbed_state_others": self.filtered_post_perturbed_state_others,
        }
        return results