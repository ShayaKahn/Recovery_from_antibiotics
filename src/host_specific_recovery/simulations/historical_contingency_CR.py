import random
import numpy as np
from src.host_specific_recovery.simulations.customer_resource import ConsumerResource


class HC_CR:
    __slots__ = ("num_samples", "pool_size", "n_resources", "num_survived_min", "num_survived_max",
                 "delta", "final_time", "max_step", "immigration_abundance", "eta",
                 "resource_min", "resource_max", "min_rho", "max_rho", "min_mortality", "max_mortality",
                 "consumer_efficiency", "method", "multiprocess", "n_jobs",
                 "upsilon", "matrix_sampling", "sample_fixed_point",
                 "C", "G", "rho", "mu", "R_star", "S_star", "empirical_growth_consumption_correlation",
                 "filter_by_event", "num_survived_list", "R_0", "Y_0", "R_p", "Y_p", "event_not_satisfied_ind",
                 "test_idx", "r_s", "y_s", "r", "y", "R_s", "Y_s", "event_not_satisfied_ind_Y_s")

    def __init__(self, num_samples, pool_size, n_resources, num_survived_min,
                 num_survived_max, delta, final_time, max_step,
                 immigration_abundance, eta, resource_min=0.5, resource_max=2.0, min_rho=1.0,
                 max_rho=1.0, min_mortality=0.05, max_mortality=0.3, consumer_efficiency=0.5,
                 method="LSODA", multiprocess=True, n_jobs=4,
                 filter_by_event=True, upsilon=0.4, matrix_sampling="general",
                 sample_fixed_point=True):

        self._validate_inputs(num_samples, pool_size, n_resources, num_survived_min, num_survived_max,
                              delta, final_time, max_step, immigration_abundance, eta, resource_min, resource_max,
                              min_rho, max_rho, min_mortality, max_mortality, method, multiprocess,
                              n_jobs, filter_by_event, upsilon, matrix_sampling, sample_fixed_point)

        self.num_samples = num_samples
        self.pool_size = pool_size
        self.n_resources = n_resources
        self.num_survived_min = num_survived_min
        self.num_survived_max = num_survived_max

        self.delta = delta
        self.final_time = final_time
        self.max_step = max_step
        self.immigration_abundance = immigration_abundance
        self.eta = eta

        self.resource_min = resource_min
        self.resource_max = resource_max
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.min_mortality = min_mortality
        self.max_mortality = max_mortality

        self.consumer_efficiency = consumer_efficiency
        self.method = method
        self.multiprocess = multiprocess
        self.n_jobs = n_jobs
        self.filter_by_event = filter_by_event

        self.upsilon = upsilon
        self.matrix_sampling = matrix_sampling
        self.sample_fixed_point = sample_fixed_point

        self.C, self.G = self._set_liu2025_matrices()

        self.empirical_growth_consumption_correlation = (
            self._growth_consumption_empirical_correlation()
        )

        self.R_star = None
        self.S_star = None
        self.rho, self.mu = self._set_rho_and_mu()

        self.num_survived_list = self._create_num_survived_list()
        self.R_0 = self._set_initial_resources(self.num_samples)
        self.Y_0 = self._set_initial_species()

        self.R_p, self.Y_p, self.event_not_satisfied_ind = self._apply_CR(
            self.R_0,
            self.Y_0,
            consumption_matrix=self.C,
            n_samples=self.num_samples,
            n_jobs=self.n_jobs,
        )

        self.Y_p = self._threshold_and_normalize(self.Y_p)

        self.test_idx, r_s, y_s, self.r, self.y = self._define_test_index_and_y_s()
        self.r_s = r_s.squeeze()
        self.y_s = self._remove_low_abundances(y_s).squeeze()

        original_test_idx = self.test_idx

        self.R_p = np.delete(self.R_p, self.event_not_satisfied_ind, axis=0)
        self.Y_p = np.delete(self.Y_p, self.event_not_satisfied_ind, axis=0)

        self.test_idx = self._compressed_index(
            self.test_idx,
            self.event_not_satisfied_ind,
        )

        self._modify_num_survived_list(original_test_idx)

        R_s_initial, Y_s_initial = self._insert_total_pool_others()

        self.R_s, self.Y_s, self.event_not_satisfied_ind_Y_s = self._apply_CR(
            R_s_initial,
            Y_s_initial,
            consumption_matrix=self.C,
            n_samples=Y_s_initial.shape[0],
            n_jobs=self.n_jobs,
        )

        self.R_s = np.delete(self.R_s, self.event_not_satisfied_ind_Y_s, axis=0)
        self.Y_s = np.delete(self.Y_s, self.event_not_satisfied_ind_Y_s, axis=0)
        self.Y_s = self._remove_low_abundances(self.Y_s)

    @staticmethod
    def _validate_inputs(num_samples, pool_size, n_resources, num_survived_min,
                         num_survived_max, delta, final_time, max_step, immigration_abundance,
                         eta, resource_min, resource_max, min_rho, max_rho, min_mortality,
                         max_mortality, method, multiprocess, n_jobs,
                         filter_by_event, upsilon, matrix_sampling, sample_fixed_point):

        if not all(isinstance(x, int) for x in [
            num_samples, pool_size, n_resources,
            num_survived_min, num_survived_max,
        ]):
            raise ValueError("sample/species/resource counts must be integers.")
        if num_samples <= 0 or pool_size <= 1 or n_resources <= 0:
            raise ValueError("num_samples, pool_size, and n_resources must be positive.")
        if num_survived_min <= 0 or num_survived_max <= 0:
            raise ValueError("num_survived_min/max must be positive.")
        if num_survived_max >= pool_size:
            raise ValueError("num_survived_max must be smaller than pool_size.")
        if num_survived_min > num_survived_max:
            raise ValueError("num_survived_min must be <= num_survived_max.")
        if not (0 < delta < 1):
            raise ValueError("delta must be between 0 and 1.")
        if final_time <= 0 or not (0 < max_step < final_time):
            raise ValueError("final_time must be positive and max_step < final_time.")
        if not (0 < immigration_abundance < 1):
            raise ValueError("immigration_abundance must be between 0 and 1.")
        if not (0 < eta < 1):
            raise ValueError("eta must be between 0 and 1.")
        if resource_min < 0 or resource_min > resource_max:
            raise ValueError("invalid resource_min/resource_max.")
        if min_rho < 0 or min_rho > max_rho:
            raise ValueError("invalid min_rho/max_rho.")
        if min_mortality < 0 or min_mortality > max_mortality:
            raise ValueError("invalid min_mortality/max_mortality.")
        if method not in ["RK45", "BDF", "RK23", "Radau", "LSODA", "DOP853"]:
            raise ValueError("invalid ODE method.")
        if not isinstance(multiprocess, bool):
            raise ValueError("multiprocess must be bool.")
        if not (isinstance(n_jobs, int) or n_jobs is None):
            raise ValueError("n_jobs must be int or None.")
        if not isinstance(filter_by_event, bool):
            raise ValueError("filter_by_event must be bool.")
        if not (0 <= upsilon <= 1):
            raise ValueError("upsilon must be between 0 and 1.")
        if matrix_sampling not in ["iid", "general"]:
            raise ValueError("matrix_sampling must be 'iid' or 'general'.")
        if not isinstance(sample_fixed_point, bool):
            raise ValueError("sample_fixed_point must be bool.")

    def _set_liu2025_matrices(self):
        G = np.random.uniform(0.0, 1.0, size=(self.n_resources, self.pool_size))
        C0 = np.random.uniform(0.0, 1.0, size=(self.n_resources, self.pool_size))

        C = self.upsilon * G + np.sqrt(1.0 - self.upsilon ** 2) * C0

        if self.matrix_sampling == "general":
            left_C = np.diag(np.random.uniform(0.01, 1.0, self.n_resources))
            right_C = np.diag(np.random.uniform(0.01, 1.0, self.pool_size))
            left_G = np.diag(np.random.uniform(0.01, 1.0, self.n_resources))
            right_G = np.diag(np.random.uniform(0.01, 1.0, self.pool_size))

            C = left_C @ C @ right_C
            G = left_G @ G @ right_G

        return C, G

    def _growth_consumption_empirical_correlation(self):
        return float(np.corrcoef(self.C.flatten(), self.G.flatten())[0, 1])

    def _set_rho_and_mu(self):
        if self.sample_fixed_point:
            self.R_star = np.random.uniform(0.01, 1.0, size=self.n_resources)
            self.S_star = np.random.uniform(0.01, 1.0, size=self.pool_size)

            mu = self.G.T @ self.R_star
            rho = self.R_star * (self.C @ self.S_star)

            return rho, mu

        rho = np.random.uniform(self.min_rho, self.max_rho, self.n_resources)
        mu = np.random.uniform(self.min_mortality, self.max_mortality, self.pool_size)

        return rho, mu

    def _create_num_survived_list(self):
        if self.num_survived_max > self.num_survived_min:
            return np.random.randint(
                self.num_survived_min,
                self.num_survived_max + 1,
                self.num_samples,
            )
        return np.full(self.num_samples, self.num_survived_min, dtype=int)

    def _set_initial_resources(self, n_samples):
        if self.sample_fixed_point and self.R_star is not None:
            noise = np.random.uniform(0.95, 1.05, size=(n_samples, self.n_resources))#np.random.uniform(0., 2., size=(n_samples, self.n_resources))
            return self.R_star[None, :] * noise

        return np.random.uniform(
            self.resource_min,
            self.resource_max,
            size=(n_samples, self.n_resources),
        )

    def _set_initial_species(self):
        Y_0 = np.zeros((self.num_samples, self.pool_size))

        for sample_idx in range(self.num_samples):
            survived_species = random.sample(
                range(self.pool_size),
                self.num_survived_list[sample_idx],
            )

            if self.sample_fixed_point and self.S_star is not None:
                Y_0[sample_idx, survived_species] = self.S_star[survived_species]
            else:
                Y_0[sample_idx, survived_species] = np.random.rand(
                    self.num_survived_list[sample_idx]
                )

        return Y_0

    def _apply_CR(self, initial_resources, initial_species, consumption_matrix, n_samples, n_jobs):
        return self._apply_CR_with_matrices(
            initial_resources,
            initial_species,
            consumption_matrix,
            self.G,
            n_samples,
            n_jobs,
        )

    def _apply_CR_with_matrices(self, initial_resources, initial_species, consumption_matrix, growth_matrix,
                                n_samples, n_jobs):
        cr_object = ConsumerResource(n_samples, self.n_resources, self.pool_size, self.delta, self.rho,
                                            self.mu, consumption_matrix, growth_matrix, initial_resources,
                                            initial_species,
                                            self.final_time, self.max_step, normalize_species=False, method=self.method,
                                            multiprocess=self.multiprocess, n_jobs=n_jobs,
                                            filter_by_event=self.filter_by_event)

        result = cr_object.solve()

        #for sample_index in range(n_samples):
        #    cr_object.plot_trajectories(sample_index=sample_index, show=True)

        return result

    def _generate_y_s(self, r, y):
        return self._apply_CR(
            r[None, :], y[None, :],
            consumption_matrix=self.C,
            n_samples=1,
            n_jobs=None,
        )

    def _define_test_index_and_y_s(self):
        event_satisfied = np.setdiff1d(
            np.arange(self.num_samples),
            self.event_not_satisfied_ind,
        )

        for idx in event_satisfied:
            r, y = self._insert_total_pool_test(idx)

            r_s, y_s, event_not_satisfied_ind_post = self._generate_y_s(r, y)

            if not event_not_satisfied_ind_post:
                return idx, r_s, y_s, r, y

        raise ValueError("The steady-state condition was not satisfied for any sample.")

    def _modify_num_survived_list(self, original_test_idx):
        removed = np.hstack([self.event_not_satisfied_ind, original_test_idx])

        self.num_survived_list = [
            item
            for idx, item in enumerate(self.num_survived_list)
            if idx not in removed
        ]

    def _insert_total_pool_test(self, test_idx):
        r = self.R_p[test_idx, :].copy()
        y = self.Y_p[test_idx, :].copy()

        mask = y == 0
        y[mask] = self.immigration_abundance

        return r, y

    def _insert_total_pool_others(self):
        R = np.delete(self.R_p.copy(), self.test_idx, axis=0)
        Y = np.delete(self.Y_p.copy(), self.test_idx, axis=0)

        for y in Y:
            y[y == 0] = self.immigration_abundance

        return R, Y

    @staticmethod
    def _compressed_index(idx, removed_indices):
        removed_indices = np.asarray(removed_indices)
        return idx - np.sum(removed_indices < idx)

    def _remove_low_abundances(self, post):
        post_copy = post.copy()
        post_copy[post_copy < self.eta] = 0.0
        return self._normalize_cohort(post_copy)

    def _threshold_and_normalize(self, Y):
        Y_copy = Y.copy()
        Y_copy[Y_copy < self.eta] = 0.0
        return self._normalize_cohort(Y_copy)

    @staticmethod
    def _normalize_cohort(cohort):
        if cohort.ndim == 1:
            total = cohort.sum()
            if total > 0:
                return cohort / total
            return cohort

        row_sums = cohort.sum(axis=1, keepdims=True)

        return np.divide(
            cohort,
            row_sums,
            out=np.zeros_like(cohort),
            where=row_sums != 0,
        )

    def get_results(self):
        return {
            "R_0": self.R_0,
            "Y_0": self.Y_0,
            "R_p": self.R_p,
            "Y_p": self.Y_p,
            "r_s": self.r_s,
            "y_s": self.y_s,
            "R_s": self.R_s,
            "Y_s": self.Y_s,
            "C": self.C,
            "G": self.G,
            "rho": self.rho,
            "mu": self.mu,
            "R_star": self.R_star,
            "S_star": self.S_star,
            "upsilon": self.upsilon,
            "matrix_sampling": self.matrix_sampling,
            "sample_fixed_point": self.sample_fixed_point,
            "empirical_growth_consumption_correlation": (
                self.empirical_growth_consumption_correlation
            ),
        }
