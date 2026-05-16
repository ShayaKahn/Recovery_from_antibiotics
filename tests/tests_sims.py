from src.host_specific_recovery.simulations.historical_contingency import HC
from unittest import TestCase
import numpy as np


class TestHC(TestCase):
    def setUp(self) -> None:
        self.num_samples = 10
        self.pool_size = 50
        self.num_survived_min = 25
        self.num_survived_max = 25
        self.mean = 0
        self.sigma = 15
        self.c = 0.05
        self.delta = 1e-4
        self.final_time = 1000
        self.max_step = 0.05
        self.epsilon = 1e-4
        self.phi = 1e-4
        self.min_growth = 1
        self.max_growth = 1
        self.symmetric = True
        self.alpha = None
        self.method = 'RK45'
        self.multiprocess = True#False
        self.switch_off = False

        # No switch off
        self.HC_no_switch = HC(self.num_samples, self.pool_size, self.num_survived_min, self.num_survived_max, self.mean,
                               self.sigma, self.c, self.delta, self.final_time, self.max_step, self.epsilon,
                               self.phi, self.min_growth, self.max_growth, self.symmetric, self.alpha, self.method,
                               self.multiprocess)

        print(self.HC_no_switch.test_idx)

        # Switch off
        self.HC_switch = HC(self.num_samples, self.pool_size, self.num_survived_min, self.num_survived_max, self.mean,
                            self.sigma, self.c, self.delta, self.final_time, self.max_step, self.epsilon,
                            self.phi, self.min_growth, self.max_growth, self.symmetric, self.alpha, self.method,
                            self.multiprocess, switch_off=True)

    def test_num_survived_lst(self):
        self.assertEqual(len(self.HC_no_switch.num_survived_list), self.num_samples -
                         len(self.HC_no_switch.event_not_satisfied_ind) - len([self.HC_no_switch.test_idx]))
        self.assertEqual(len(self.HC_switch.num_survived_list), self.num_samples -
                         len(self.HC_switch.event_not_satisfied_ind) - len([self.HC_switch.test_idx]))
        if self.num_survived_min == self.num_survived_max:
            self.assertEqual(np.max(self.HC_no_switch.num_survived_list), self.num_survived_min)
            self.assertEqual(np.min(self.HC_switch.num_survived_list), self.num_survived_min)
        else:
            self.assertTrue(np.max(self.HC_no_switch.num_survived_list) <= self.num_survived_max)
            self.assertTrue(np.min(self.HC_no_switch.num_survived_list) >= self.num_survived_min)
            self.assertTrue(np.max(self.HC_switch.num_survived_list) <= self.num_survived_max)
            self.assertTrue(np.min(self.HC_switch.num_survived_list) >= self.num_survived_min)

    def test_interaction_matrix(self):
        self.assertTrue(np.array_equal(np.diag(self.HC_no_switch.A), np.zeros((1, self.pool_size)).squeeze()))
        self.assertTrue(np.array_equal(np.diag(self.HC_switch.A), np.zeros((1, self.pool_size)).squeeze()))

    def test_set_logistic_growth(self):
        self.assertEqual(self.HC_no_switch.s.shape, (self.pool_size,))

    def test_set_growth_rate(self):
        self.assertEqual(self.HC_no_switch.r.shape, (self.pool_size,))

    def test_set_initial_conditions(self):
        self.assertEqual(self.HC_no_switch.Y_0.shape[0], self.num_samples)
        self.assertEqual(self.HC_switch.Y_0.shape[0], self.num_samples)
        self.assertTrue(np.max(self.HC_no_switch.Y_0.astype(bool).sum(axis=1)) <= self.num_survived_max)
        self.assertTrue(np.min(self.HC_no_switch.Y_0.astype(bool).sum(axis=1)) >= self.num_survived_min)
        self.assertTrue(np.max(self.HC_switch.Y_0.astype(bool).sum(axis=1)) <= self.num_survived_max)
        self.assertTrue(np.min(self.HC_switch.Y_0.astype(bool).sum(axis=1)) >= self.num_survived_min)

    def test_set_symmetric_interaction_matrix(self):
        N = self.HC_no_switch._set_symmetric_interaction_matrix()
        mat = N.copy()
        index = np.where(mat != 0)
        mat[index] = 1
        self.assertTrue(np.array_equal(mat, mat.T))

    def test_insert_total_pool_others(self):
        mat = self.HC_switch.y
        self.assertAlmostEqual(np.min(mat), self.epsilon, places=5)

    def test_consistency(self):
        event_not_satisfied_ind = self.HC_no_switch.event_not_satisfied_ind
        event_not_satisfied_ind_Y_s = self.HC_no_switch.event_not_satisfied_ind_Y_s
        results = self.HC_no_switch.get_results()
        Y_s = results["Y_s"]
        self.assertEqual(Y_s.shape[0],
                         self.num_samples - len(event_not_satisfied_ind) - len(event_not_satisfied_ind_Y_s) - 1)
        