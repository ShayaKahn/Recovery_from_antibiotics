import unittest

import numpy as np

from src.host_specific_recovery.analysis.new_species_first_detection_analysis import (
    run_new_species_first_detection_analysis,
)


class TestNewSpeciesFirstDetectionAnalysis(unittest.TestCase):
    def test_counts_first_detections_including_last_timepoint(self):
        dataset = {
            "baseline": [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            "abx": [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
            ],
            "post_abx_cohorts": [
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                ],
                [
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 1, 0],
                ],
            ],
            "times_post_abx": [5, 12, 30],
            "filtered_keys": ["subject-a", "subject-b"],
        }

        outputs = run_new_species_first_detection_analysis(dataset)

        np.testing.assert_array_equal(
            outputs["subject_ids"],
            ["subject-a", "subject-b"],
        )
        np.testing.assert_array_equal(outputs["times_post_abx"], [5, 12, 30])
        np.testing.assert_array_equal(
            outputs["new_species_counts"],
            [
                [1, 1, 1],
                [1, 0, 1],
            ],
        )
        np.testing.assert_array_equal(
            outputs["total_new_species_counts"],
            [3, 2],
        )
        np.testing.assert_array_equal(
            outputs["total_new_species_counts_excluding_last"],
            [2, 1],
        )

    def test_counts_colonized_species_and_excludes_final_first_detections(self):
        dataset = {
            "baseline": [[0, 0, 0, 0]],
            "abx": [[0, 0, 0, 0]],
            "post_abx_cohorts": [
                [[1, 1, 0, 0]],
                [[1, 0, 1, 0]],
                [[1, 0, 1, 1]],
            ],
            "times_post_abx": [5, 12, 30],
        }

        outputs = run_new_species_first_detection_analysis(dataset)

        np.testing.assert_array_equal(outputs["new_species_counts"], [[2, 1, 1]])
        np.testing.assert_array_equal(outputs["total_new_species_counts"], [4])
        np.testing.assert_array_equal(
            outputs["total_new_species_counts_excluding_last"],
            [3],
        )
        np.testing.assert_array_equal(outputs["colonized_new_species_counts"], [2])

    def test_rejects_mismatched_timepoints(self):
        dataset = {
            "baseline": [[0]],
            "abx": [[0]],
            "post_abx_cohorts": [[[1]], [[0]]],
            "times_post_abx": [5],
        }

        with self.assertRaisesRegex(ValueError, "one value per post-ABX timepoint"):
            run_new_species_first_detection_analysis(dataset)


if __name__ == "__main__":
    unittest.main()
