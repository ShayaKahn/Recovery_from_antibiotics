from unittest import TestCase
import numpy as np
from src.host_specific_recovery.metrics.similarity import Similarity


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
        #This function tests the Overlap similarity method for sample and matrix cases.
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
        #This function tests the Dice similarity method for sample and matrix cases.
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Dice', norm=True)
        dice_val = similarity_smp.calculate_similarity()
        self.assertEqual(dice_val, 8/9)
        # matrix
        similarity_mat = Similarity(self.first_sample, self.matrix, method='Dice', norm=True)
        dice_val_mat = similarity_mat.calculate_similarity()
        self.assertListEqual(list(dice_val_mat), [8/9, 4/9, 4/9, 4/7])

    def test_szymkiewicz_simpson(self):
        #This function tests the Szymkiewicz Simpson similarity method for sample and matrix cases.
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
        # This function tests the Recovery similarity method for sample and matrix cases.
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
        # This function tests the Specificity similarity method for sample and matrix cases.
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
        # This function tests the Weighted Jaccard similarity method for sample and matrix cases.
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
        # This function tests the Weighted Jaccard symmetric similarity method.
        # sample
        similarity_smp = Similarity(self.first_sample, self.sample, method='Weighted Jaccard symmetric',
                                    norm=False)
        jaccard_w_val = similarity_smp.calculate_similarity()
        self.assertEqual(jaccard_w_val, 0.5333333333333333)
