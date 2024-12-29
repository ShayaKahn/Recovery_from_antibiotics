from methods.null_model import NullModel
from data_processing.rarify import Rarify
from methods.surrogate import Surrogate
from methods.similarity_correlation import SimilarityCorrelation
from methods.similarity import Similarity
from data_processing.optimal import OptimalCohort
from unittest import TestCase
import numpy as np
import pandas as pd

class TestSimilarityCorrelation(TestCase):
    """
    This class tests the SimilarityCorrelation class.
    """
    def setUp(self) -> None:
        keys = ["A", "B", "C", "D", "E"]
        abx = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
        base = np.array([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                         [1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                         [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                         [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]])
        self.post_ABX_container = {"A": np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                                  [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                                                  [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]]),
                                   "B": np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                                                  [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]),
                                   "C": np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                                                  [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                                  [0, 1, 0, 1, 1, 1, 0, 1, 1, 1]]),
                                   "D": np.array([[1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                                  [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                                                  [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]]),
                                   "E": np.array([[1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                                                  [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
                                                  [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])
                                   }
        keys_ref = ["A", "B", "C", "D", "E", "F", "G"]
        base_others = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
        base_ref = np.vstack([base, base_others])
        self.baseline_ref = pd.DataFrame(base_ref.T, columns=keys_ref)
        self.ABX = pd.DataFrame(abx.T, columns=keys)
        self.baseline = pd.DataFrame(base.T, columns=keys)
        self.method = "Jaccard"
        self.timepoints = 1
        self.iters = 10
        self.new = False
        self.sim = SimilarityCorrelation(self.ABX, self.baseline, self.post_ABX_container, self.baseline_ref,
                                         self.method, self.timepoints, self.iters, self.new, True)
        self.sim_soft = SimilarityCorrelation(self.ABX, self.baseline, self.post_ABX_container, self.baseline_ref,
                                              self.method, self.timepoints, self.iters, self.new, False)
        self.timepoints_vals = self.sim._returned_species(self.sim.baseline[0, :], self.sim.ABX[0, :],
                                                          self.sim.post_ABX_matrices[0])
        self.timepoints_vals_soft = self.sim_soft._returned_species(self.sim.baseline[1, :], self.sim.ABX[1, :],
                                                                    self.sim.post_ABX_matrices[1])

    def test_generate_synthetic_cohorts(self):
        cohorts_lst = self.sim.synthetic_baseline_lst
        for cohort in cohorts_lst:
            self.assertTrue(np.array_equal(cohort.sum(axis=1), self.baseline_ref.to_numpy().T.sum(axis=1)))

    def test_returned_species(self):
        # strict
        self.assertEqual(len(self.timepoints_vals), self.post_ABX_container["A"].shape[0])
        self.assertTrue(np.array_equal(self.timepoints_vals[0], np.array([False, False, False, False, False, False,
                                                                          True, False, False, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals[1], np.array([False, False, False, False, True, False,
                                                                          False, False, True, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals[2], np.array([False, False, False, False, False, False,
                                                                          False, False, False, False])))
        # soft
        self.assertEqual(len(self.timepoints_vals_soft), self.post_ABX_container["B"].shape[0])
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[0], np.array([False, False, True, False, False, False,
                                                                               False, False, True, False])))
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[1], np.array([False, True, False, False, False, False,
                                                                               False, False, False, True])))
        self.assertTrue(np.array_equal(self.timepoints_vals_soft[2], np.array([False, False, False, False, False, True,
                                                                               False, False, False, False])))

    def test_find_subset(self):
        idx = self.sim._find_subset(self.sim.baseline[0, :], self.sim.ABX[0, :], self.sim.post_ABX_matrices[0],
                                    self.timepoints_vals)
        self.assertEqual(np.size(idx), self.post_ABX_container["A"].shape[0] - self.timepoints)


class TestOptimalCohort(TestCase):
    """
    This class tests the OptimalCohort class.
    """
    def setUp(self) -> None:
        # Two default samples.
        self.samples_dict = {'a': np.array([[11, 0, 8], [3, 9, 2], [0, 1, 3]]),
                             'b': np.array([[7, 1, 2], [1, 6, 0], [2, 3, 8], [8, 2, 5], [0, 1, 0]]),
                             'c': np.array([[35, 0, 17], [3, 4, 3], [1, 0, 8]]),
                             'd': np.array([[12, 7, 4], [1, 0, 0], [7, 1, 0], [6, 6, 6]])}
        self.optimal = OptimalCohort(self.samples_dict, method='jaccard')
        self.optimal_samples = self.optimal.get_optimal_samples()
    def test_get_optimal_samples(self):
        self.assertEqual(np.sum(self.optimal_samples[0], axis=1).tolist(),
                         np.ones(np.size(self.optimal_samples[0], axis=0)).tolist())

class Test_Rarify(TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame({
                  'A': [1, 3, 0, 2],
                  'B': [2, 0, 2, 1],
                  'C': [0, 4, 0, 0],
                  'D': [50, 12, 0, 0],
                   })
        self.min = Rarify(self.df)
        self.depth = Rarify(self.df, depth=5)

    def test_rarify(self):
        rar_df_min = self.min.rarify()
        rar_df_depth = self.depth.rarify()

class Test_Similarity(TestCase):
    def setUp(self) -> None:
        self.first_sample = np.array([[0.1, 0, 0.2, 0.4, 0, 0, 0.1, 0.3]])
        self.sample = np.array([[0.1, 0, 0.2, 0.7, 0, 0, 0.2, 0]])
        self.matrix = np.array([[0.2, 0, 0.2, 0.5, 0, 0, 0.2, 0],
                                [0, 0.2, 0.2, 0, 0, 0.5, 0.2, 0],
                                [0, 0, 0.3, 0, 0.5, 0.1, 0.2, 0],
                                [0.9, 0, 0, 0, 0, 0, 0, 0.2]])

    def test_jaccard(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Jaccard', norm=True)
        jaccard_val = similarity_smp.calculate_similarity()
        self.assertEqual(jaccard_val, 0.8)
        self.assertEqual(similarity_smp.sample_first.sum(), 1)
        self.assertEqual(similarity_smp.matrix.sum(), 1)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Jaccard', norm=True)
        jaccard_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(jaccard_val_mat), [0.8, 2/7, 2/7, 0.4])
        self.assertEqual(similarity_mat.sample_first.sum(), 1.)
        self.assertTrue(np.allclose(list(similarity_mat.matrix.sum(axis=1)), list(np.ones((4,)))))

    def test_overlap(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Overlap', norm=False)
        overlap_val = similarity_smp.calculate_similarity()
        self.assertEqual(overlap_val, 1)
        self.assertNotEqual(similarity_smp.sample_first.sum(), 1)
        self.assertNotEqual(similarity_smp.matrix.sum(), 1)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Overlap', norm=False)
        overlap_val_mat = similarity_mat.calculate_similarity()
        self.assertTrue(np.allclose(list(overlap_val_mat), [0.95, 0.35, 0.4, 0.75]))
        self.assertNotEqual(similarity_mat.sample_first.sum(), 1.)
        self.assertFalse(np.allclose(list(similarity_mat.matrix.sum(axis=1)), [1, 1, 1, 1]))

    def test_dice(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Dice', norm=True)
        dice_val = similarity_smp.calculate_similarity()
        self.assertEqual(dice_val, 8/9)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Dice', norm=True)
        dice_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(dice_val_mat), [8/9, 4/9, 4/9, 4/7])

    def test_szymkiewicz_simpson(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Szymkiewicz Simpson',
                                    norm=True)
        ss_val = similarity_smp.calculate_similarity()
        self.assertEqual(ss_val, 1)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Szymkiewicz Simpson',
                                    norm=True)
        ss_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(ss_val_mat), [1, 0.5, 0.5, 1])

    def test_recovery(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Recovery',
                                    norm=True)
        recovery_val = similarity_smp.calculate_similarity()
        self.assertEqual(recovery_val, 0.8)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Recovery',
                                    norm=True)
        recovery_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(recovery_val_mat), [0.8, 0.4, 0.4, 0.4])

    def test_specificity(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Specificity',
                                    norm=True)
        specificity_val = similarity_smp.calculate_similarity()
        self.assertEqual(specificity_val, 1)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Specificity',
                                    norm=True)
        specificity_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(specificity_val_mat), [1, 0.5, 0.5, 1])

    def test_weighted_jaccard(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Weighted Jaccard',
                                    norm=False)
        jaccard_w_val = similarity_smp.calculate_similarity()
        self.assertEqual(jaccard_w_val, 0.8/1.1)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Weighted Jaccard',
                                    norm=False)
        jaccard_w_val_mat = similarity_mat.calculate_similarity()
        self.assertTrue(np.allclose(list(jaccard_w_val_mat), [0.8/1.1, 0.3/1.1, 0.3/1.1, 0.4/1.1]))

    def test_weighted_jaccard_symmetric(self):
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Weighted Jaccard symmetric',
                                    norm=False)
        jaccard_w_val = similarity_smp.calculate_similarity()
        self.assertEqual(jaccard_w_val, 0.5333333333333333)


class Test_NullModel(TestCase):
    def setUp(self) -> None:
        self.baseline_sample = np.array([0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0])
        self.ABX_sample = np.array([0, 0, 0.3, 0.1, 0, 0.1, 0, 0, 0, 0.5])
        self.test_matrix = np.array([[0.5, 0, 0.1, 0, 0.2, 0.2, 0, 0, 0, 0],
                                    [0.5, 0, 0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0]])
        self.baseline = np.array([[0.1, 0, 0, 0.2, 0.3, 0.1, 0, 0.1, 0.2, 0],
                                  [0.5, 0.1, 0, 0, 0.1, 0.1, 0, 0.1, 0.1, 0],
                                  [0.1, 0, 0.2, 0.1, 0, 0.1, 0, 0, 0.2, 0.3]])
        self.num_reals = 3
        self.timepoints = 1
        self.null_model = NullModel(self.baseline_sample, self.ABX_sample, self.baseline,
                                    self.test_matrix, self.num_reals, self.timepoints)

    def test_find_subset(self):
        self.assertListEqual(list(self.null_model.subset_comp), [0, 2, 4, 5])
        self.assertListEqual(list(self.null_model.subset), [1, 3, 6, 7, 8, 9])
        self.assertEqual(np.size(self.null_model.subset_comp) + np.size(self.null_model.subset), self.baseline.shape[1])

    def test_distance(self):
        # Test the constraint
        self.assertEqual(np.size(np.nonzero(self.null_model.synthetic_samples[0, self.null_model.subset])),
                         np.size(np.nonzero(self.test_matrix[-1, :][self.null_model.subset])))
        self.assertEqual(self.null_model.distance(method="Specificity")[0], 0)


class Test_Surrogate(TestCase):
    def setUp(self) -> None:
        self.test_post_abx_matrix = np.array([[1, 0, 1, 0, 0, 0, 0, 1],
                                              [1, 0, 1, 1, 0, 1, 0, 1],
                                              [1, 0, 0, 0, 0, 0, 1, 1],
                                              [1, 1, 1, 1, 1, 1, 1, 0]])
        self.test_base_samples_collection = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1]])}
        self.test_base_samples_collection_mat = {"Test subject": np.array([[1, 0, 1, 1, 1, 0, 1, 1],
                                                                           [1, 0, 1, 0, 1, 0, 1, 1],
                                                                           [1, 0, 1, 1, 1, 1, 0, 1]])}
        self.base_samples_collections = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0]]),
                                         "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1]])}
        self.base_samples_collections_mat = {"Subject A": np.array([[0, 1, 1, 1, 0, 1, 1, 0],
                                                                    [0, 1, 1, 1, 0, 1, 0, 0]]),
                                             "Subject B": np.array([[1, 1, 0, 1, 1, 0, 0, 1],
                                                                    [1, 1, 0, 1, 1, 0, 0, 0]])}
        self.test_abx_sample = np.array([[1, 1, 0, 0, 0, 0, 0, 1]])
        self.timepoints = 1
        self.surrogate = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                   self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints)
        self.results = self.surrogate.apply_surrogate_data_analysis()
        self.surrogate_not_strict = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                          self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints,
                                          strict=False)
        self.results_not_strict = self.surrogate_not_strict.apply_surrogate_data_analysis()
        self.surrogate_null = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                        self.test_post_abx_matrix, self.test_abx_sample, timepoints=0)
        self.results_null = self.surrogate_null.apply_surrogate_data_analysis()
        self.surrogate_null_not_strict = Surrogate(self.base_samples_collections, self.test_base_samples_collection,
                                                   self.test_post_abx_matrix, self.test_abx_sample, timepoints=0,
                                                   strict=False)
        self.results_null_not_strict = self.surrogate_null_not_strict.apply_surrogate_data_analysis()
        self.surrogate_mat = Surrogate(self.base_samples_collections_mat, self.test_base_samples_collection_mat,
                                       self.test_post_abx_matrix, self.test_abx_sample, timepoints=self.timepoints)
        self.results_mat = self.surrogate_mat.apply_surrogate_data_analysis()

    def test_find_subset(self):
        self.assertListEqual(list(self.surrogate.subset), [2, 3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_not_strict.subset), [3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_null.subset), [2, 3, 4, 5, 6, 7])
        self.assertListEqual(list(self.surrogate_null_not_strict.subset), [2, 3, 4, 5, 6, 7])

    def test_matrix_case(self):
        self.assertEqual(self.surrogate_mat.test_key, "Test subject")
        self.assertListEqual(list(self.surrogate_mat.test_base_sample), [1, 0, 1, 1, 1, 0, 1, 1])