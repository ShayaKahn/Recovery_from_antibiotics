import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.host_specific_recovery.analysis.returned_species_analysis import (
    ReturnedSpeciesAnalysis,
)
from src.host_specific_recovery.visualizations.plot_returned_species_analysis import (
    plot_returned_species_analysis,
)


class TestReturnedSpeciesAnalysis(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "baseline": np.array([
                [0.10, 0.20, 0.30, 0.40, 0.00, 0.15],
                [0.00, 0.10, 0.20, 0.30, 0.40, 0.00],
            ]),
            "abx": np.array([
                [0.00, 0.00, 0.01, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            ]),
            "post_abx_cohorts": [
                np.array([
                    [0.00, 0.02, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ]),
                np.array([
                    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.05, 0.00, 0.00],
                ]),
                np.array([
                    [0.05, 0.04, 0.00, 0.00, 0.20, 0.03],
                    [0.00, 0.03, 0.04, 0.00, 0.02, 0.00],
                ]),
            ],
            "taxa_ids": ["a", "b", "c", "d", "e", "f"],
            "subject_ids": ["subject-1", "subject-2"],
            "times_post_abx": [4, 10, 30],
            "depth": 10_000,
        }

    def test_selects_only_taxa_first_detected_at_last_follow_up(self):
        outputs = ReturnedSpeciesAnalysis(self.dataset).run("subject-1")

        np.testing.assert_array_equal(outputs["species_ids"], ["a", "f"])
        np.testing.assert_array_equal(outputs["species_indices"], [0, 5])
        np.testing.assert_allclose(
            outputs["baseline_relative_abundance"],
            [0.10, 0.15],
        )
        np.testing.assert_allclose(
            outputs["last_follow_up_relative_abundance"],
            [0.05, 0.03],
        )
        np.testing.assert_allclose(
            outputs["relative_abundances"],
            [
                [0.10, 0.00, 0.00, 0.00, 0.05],
                [0.15, 0.00, 0.00, 0.00, 0.03],
            ],
        )
        np.testing.assert_array_equal(
            outputs["timeline_labels"],
            ["Baseline", "ABX", "Day 4", "Day 10", "Day 30"],
        )
        self.assertAlmostEqual(outputs["detection_threshold"], 3 / 10_000)

        summary = outputs["returned_species"]
        self.assertEqual(
            list(summary.columns),
            ReturnedSpeciesAnalysis.SUMMARY_COLUMNS,
        )
        self.assertEqual(summary["subject_id"].tolist(), ["subject-1", "subject-1"])
        self.assertEqual(outputs["n_first_returned_species"], 0)
        self.assertEqual(outputs["first_returned_relative_abundances"].shape, (0, 5))

    def test_first_returned_taxa_must_persist_in_every_post_abx_sample(self):
        dataset = {
            "baseline": np.array([[0.2, 0.1, 0.3]]),
            "abx": np.array([[0.0, 0.0, 0.01]]),
            "post_abx_cohorts": [
                np.array([[0.02, 0.03, 0.02]]),
                np.array([[0.04, 0.00, 0.02]]),
                np.array([[0.05, 0.01, 0.02]]),
            ],
            "taxa_ids": ["persistent", "later-missing", "present-during-abx"],
            "subject_ids": ["subject-1"],
            "times_post_abx": [5, 10, 20],
            "depth": 1_000,
        }

        outputs = ReturnedSpeciesAnalysis(dataset).run("subject-1")

        self.assertEqual(outputs["n_first_returned_species"], 1)
        np.testing.assert_array_equal(
            outputs["first_returned_species_ids"],
            ["persistent"],
        )
        np.testing.assert_allclose(
            outputs["first_returned_relative_abundances"],
            [[0.2, 0.0, 0.02, 0.04, 0.05]],
        )

    def test_rejects_unknown_subject(self):
        analysis = ReturnedSpeciesAnalysis(self.dataset)
        with self.assertRaisesRegex(KeyError, "Unknown subject_id"):
            analysis.run("not-a-subject")

    def test_selects_qualifying_taxa_for_another_subject(self):
        outputs = ReturnedSpeciesAnalysis(self.dataset).run("subject-2")

        self.assertEqual(outputs["n_returned_species"], 3)
        np.testing.assert_array_equal(outputs["species_ids"], ["b", "c", "e"])

    def test_returns_empty_outputs_when_no_taxa_qualify(self):
        dataset = dict(self.dataset)
        dataset["post_abx_cohorts"] = [
            cohort.copy() for cohort in self.dataset["post_abx_cohorts"]
        ]
        dataset["post_abx_cohorts"][-1][:] = 0

        outputs = ReturnedSpeciesAnalysis(dataset).run("subject-1")

        self.assertEqual(outputs["n_returned_species"], 0)
        self.assertTrue(outputs["returned_species"].empty)
        self.assertEqual(outputs["relative_abundances"].shape, (0, 5))

    def test_non_rarefied_analysis_uses_sample_specific_depths(self):
        dataset = dict(self.dataset)
        dataset["baseline_no_rar"] = self.dataset["baseline"].copy()
        dataset["abx_no_rar"] = self.dataset["abx"].copy()
        dataset["post_abx_cohorts_no_rar"] = [
            cohort.copy() for cohort in self.dataset["post_abx_cohorts"]
        ]
        dataset["sample_depth"] = pd.DataFrame(
            [
                [1_000, 2_000, 3_000, 4_000, 5_000, 6_000],
                [1_500, 2_500, 3_500, 4_500, 5_500, 6_500],
            ],
            index=["subject-1", "subject-2"],
            columns=[
                "baseline",
                "abx_7",
                "abx",
                "post_abx_4",
                "post_abx_10",
                "post_abx",
            ],
        )

        outputs = ReturnedSpeciesAnalysis(
            dataset,
            non_rarefied=True,
        ).run("subject-1")

        np.testing.assert_array_equal(outputs["species_ids"], ["a", "f"])
        np.testing.assert_allclose(
            outputs["sequencing_depths"],
            [1_000, 3_000, 4_000, 5_000, 6_000],
        )
        np.testing.assert_array_equal(
            outputs["sample_depth_timepoints"],
            ["baseline", "abx", "post_abx_4", "post_abx_10", "post_abx"],
        )
        np.testing.assert_allclose(
            outputs["one_read_abundances"],
            1 / np.array([1_000, 3_000, 4_000, 5_000, 6_000]),
        )
        np.testing.assert_allclose(
            outputs["detection_thresholds"],
            3 / np.array([1_000, 3_000, 4_000, 5_000, 6_000]),
        )
        self.assertTrue(outputs["non_rarefied"])

    def test_selects_survived_taxa_reduced_more_than_100_fold(self):
        dataset = {
            "baseline": np.array([[0.5, 0.2, 0.1, 0.1]]),
            "abx": np.array([[0.004, 0.002, 0.001, 0.01]]),
            "post_abx_cohorts": [
                np.array([[0.01, 0.01, 0.00, 0.01]]),
                np.array([[0.02, 0.02, 0.02, 0.02]]),
            ],
            "taxa_ids": ["selected", "exactly-100", "missing-post", "small-change"],
            "subject_ids": ["subject-1"],
            "times_post_abx": [5, 20],
            "depth": 1_000,
        }

        outputs = ReturnedSpeciesAnalysis(dataset).run("subject-1")

        self.assertEqual(outputs["n_returned_species"], 0)
        self.assertEqual(outputs["n_reduced_survived_species"], 1)
        np.testing.assert_array_equal(
            outputs["reduced_survived_species_ids"],
            ["selected"],
        )
        np.testing.assert_allclose(
            outputs["reduced_survived_relative_abundances"],
            [[0.5, 0.004, 0.01, 0.02]],
        )
        np.testing.assert_allclose(outputs["survived_fold_reductions"], [125])

        stricter_outputs = ReturnedSpeciesAnalysis(
            dataset,
            survived_reduction_threshold=130,
        ).run("subject-1")
        self.assertEqual(stricter_outputs["n_reduced_survived_species"], 0)
        self.assertEqual(stricter_outputs["survived_reduction_threshold"], 130)


class TestPlotReturnedSpeciesAnalysis(unittest.TestCase):
    def test_plot_shows_log_scale_undetected_floor_and_threshold(self):
        dataset = {
            "baseline": [[0.1]],
            "abx": [[0.0]],
            "post_abx_cohorts": [[[0.0]], [[0.02]]],
            "taxa_ids": ["taxon-a"],
            "subject_ids": ["subject-1"],
            "times_post_abx": [5, 20],
            "depth": 1_000,
        }
        outputs = ReturnedSpeciesAnalysis(dataset).run("subject-1")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "returned.png"
            figure, axes = plot_returned_species_analysis(outputs, path=output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(axes.get_yscale(), "log")
            self.assertEqual(axes.get_ylabel(), "Relative abundance (log scale)")
            self.assertEqual(
                [tick.get_text() for tick in axes.get_xticklabels()],
                ["Baseline", "ABX", "Day 5", "Day 20"],
            )
            taxon_line = next(
                line for line in axes.lines if line.get_label() == "taxon-a"
            )
            np.testing.assert_allclose(taxon_line.get_xdata(), [0, 10, 15, 30])
            self.assertEqual(taxon_line.get_alpha(), 0.7)
            self.assertEqual(taxon_line.get_color(), "#25478F")
            np.testing.assert_allclose(
                taxon_line.get_ydata(),
                [0.1, 0.001, 0.001, 0.02],
            )
            threshold_lines = [
                line for line in axes.lines
                if line.get_label() == "Upper boundary at 3/$N_0$: 95% conditional detection boundary"
            ]
            self.assertEqual(len(threshold_lines), 1)
            np.testing.assert_allclose(threshold_lines[0].get_ydata(), [0.003, 0.003])
            sampling_floor_lines = [
                line for line in axes.lines
                if line.get_label() == "Lower boundary at 1/$N_0$: one-read abundance"
            ]
            self.assertEqual(len(sampling_floor_lines), 1)
            np.testing.assert_allclose(sampling_floor_lines[0].get_ydata(), [0.001, 0.001])

            low_count_regions = [
                patch for patch in axes.patches if patch.get_alpha() == 0.12
            ]
            self.assertEqual(len(low_count_regions), 1)
            region_labels = [
                text for text in axes.texts
                if text.get_text() == "sampling-sensitive low-count region"
            ]
            self.assertEqual(len(region_labels), 1)
            self.assertGreater(region_labels[0].get_position()[1], 0.003)
            self.assertIsNotNone(region_labels[0].arrow_patch)

            legend_labels = [
                text.get_text() for text in axes.get_legend().get_texts()
            ]
            self.assertEqual(
                legend_labels,
                [
                    "Late returning species",
                    "Upper boundary at 3/$N_0$: 95% conditional detection boundary",
                    "Lower boundary at 1/$N_0$: one-read abundance",
                    "Undetected (shown at 1/$N_0$)",
                ],
            )
            self.assertNotIn("taxon-a", legend_labels)

            triangle_offsets = axes.collections[0].get_offsets()
            np.testing.assert_allclose(triangle_offsets[:, 1], [0.001, 0.001])
            self.assertEqual(axes.collections[0].get_alpha(), 0.7)
            plt.close(figure)

    def test_plot_uses_variable_non_rarefied_boundaries(self):
        outputs = {
            "relative_abundances": [[0.1, 0.0, 0.0, 0.02]],
            "species_ids": ["taxon-a"],
            "timeline_labels": ["Baseline", "ABX", "Day 5", "Day 20"],
            "follow_up_times": [5, 20],
            "sequencing_depths": [1_000, 2_000, 4_000, 8_000],
            "non_rarefied": True,
            "reduced_survived_relative_abundances": [[0.2, 0.001, 0.004, 0.03]],
            "first_returned_relative_abundances": [[0.15, 0.0, 0.03, 0.02]],
            "survived_reduction_threshold": 100,
        }

        figure, axes = plot_returned_species_analysis(outputs)

        upper_line = next(
            line for line in axes.lines
            if line.get_label() == "Upper boundary at 3/$N$: 95% conditional detection boundary"
        )
        lower_line = next(
            line for line in axes.lines
            if line.get_label() == "Lower boundary at 1/$N$: one-read abundance"
        )
        np.testing.assert_allclose(
            upper_line.get_ydata(),
            3 / np.array([1_000, 2_000, 4_000, 8_000]),
        )
        np.testing.assert_allclose(
            lower_line.get_ydata(),
            1 / np.array([1_000, 2_000, 4_000, 8_000]),
        )

        low_count_regions = [
            collection for collection in axes.collections
            if collection.get_alpha() == 0.12
        ]
        self.assertEqual(len(low_count_regions), 1)
        triangle_collection = next(
            collection for collection in axes.collections
            if collection.get_alpha() == 0.7
            and collection.get_offsets().shape[0] == 2
        )
        np.testing.assert_allclose(
            triangle_collection.get_offsets()[:, 1],
            [1 / 2_000, 1 / 4_000],
        )
        survived_line = next(
            line for line in axes.lines if line.get_alpha() == 0.45
        )
        np.testing.assert_allclose(
            survived_line.get_ydata(),
            [0.2, 0.001, 0.004, 0.03],
        )
        self.assertEqual(survived_line.get_color(), "#7f7f7f")
        first_returned_line = next(
            line for line in axes.lines if line.get_color() == "#4285FF"
        )
        np.testing.assert_allclose(
            first_returned_line.get_ydata(),
            [0.15, 1 / 2_000, 0.03, 0.02],
        )
        last_returned_line = next(
            line for line in axes.lines if line.get_color() == "#25478F"
        )
        np.testing.assert_allclose(
            last_returned_line.get_ydata(),
            [0.1, 1 / 2_000, 1 / 4_000, 0.02],
        )
        survived_circle_collection = next(
            collection for collection in axes.collections
            if collection.get_alpha() == 0.45
            and collection.get_offsets().shape[0] == 4
        )
        np.testing.assert_allclose(
            survived_circle_collection.get_offsets()[:, 1],
            [0.2, 0.001, 0.004, 0.03],
        )
        legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
        self.assertEqual(
            legend_labels,
            [
                "Survived species with >100-fold ABX reduction",
                "Early returning species",
                "Late returning species",
                "Upper boundary at 3/$N$: 95% conditional detection boundary",
                "Lower boundary at 1/$N$: one-read abundance",
                "Undetected (shown at 1/$N$)",
            ],
        )
        plt.close(figure)

    def test_plot_supports_linear_scale(self):
        outputs = {
            "relative_abundances": [[0.1, 0.0, 0.02]],
            "species_ids": ["taxon-a"],
            "timeline_labels": ["Baseline", "ABX", "Day 20"],
            "follow_up_times": [20],
            "sequencing_depth": 1_000,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "returned-linear.png"
            figure, axes = plot_returned_species_analysis(
                outputs,
                log_scale=False,
                path=output_path,
            )

            self.assertEqual(axes.get_yscale(), "linear")
            self.assertEqual(axes.get_ylabel(), "Relative abundance")
            self.assertEqual(axes.get_ylim()[0], 0)
            plt.close(figure)

    def test_plot_supports_custom_tick_mark_size(self):
        outputs = {
            "relative_abundances": [[0.1, 0.0, 0.02]],
            "species_ids": ["taxon-a"],
            "timeline_labels": ["Baseline", "ABX", "Day 20"],
            "follow_up_times": [20],
            "sequencing_depth": 1_000,
        }

        figure, axes = plot_returned_species_analysis(
            outputs,
            tick_length=9,
            tick_width=2.5,
        )

        for tick in [axes.xaxis.majorTicks[0], axes.yaxis.majorTicks[0]]:
            self.assertEqual(tick.tick1line.get_markersize(), 9)
            self.assertEqual(tick.tick1line.get_markeredgewidth(), 2.5)
        plt.close(figure)

    def test_plot_top_x_is_applied_independently_to_each_category(self):
        outputs = {
            "relative_abundances": [
                [0.1, 0.0, 0.0, 0.01],
                [0.1, 0.0, 0.0, 0.04],
                [0.1, 0.0, 0.0, 0.02],
            ],
            "species_ids": ["last-low", "last-high", "last-middle"],
            "timeline_labels": ["Baseline", "ABX", "Day 5", "Day 20"],
            "follow_up_times": [5, 20],
            "sequencing_depth": 1_000,
            "reduced_survived_relative_abundances": [
                [0.2, 0.001, 0.01, 0.03],
                [0.2, 0.001, 0.01, 0.06],
                [0.2, 0.001, 0.01, 0.04],
            ],
            "first_returned_relative_abundances": [
                [0.1, 0.0, 0.02, 0.05],
                [0.1, 0.0, 0.02, 0.02],
                [0.1, 0.0, 0.02, 0.07],
            ],
        }

        figure, axes = plot_returned_species_analysis(outputs, top_x=2)

        expected_final_abundances = {
            "#7f7f7f": [0.04, 0.06],
            "#4285FF": [0.05, 0.07],
            "#25478F": [0.02, 0.04],
        }
        for color, expected in expected_final_abundances.items():
            trajectory_lines = [
                line for line in axes.lines if line.get_color() == color
            ]
            self.assertEqual(len(trajectory_lines), 2)
            self.assertEqual(
                sorted(line.get_ydata()[-1] for line in trajectory_lines),
                expected,
            )
        plt.close(figure)

    def test_plot_rejects_non_positive_top_x(self):
        outputs = {
            "relative_abundances": [[0.1, 0.0, 0.02]],
            "species_ids": ["taxon-a"],
            "timeline_labels": ["Baseline", "ABX", "Day 20"],
            "follow_up_times": [20],
            "sequencing_depth": 1_000,
        }

        with self.assertRaisesRegex(ValueError, "top_x"):
            plot_returned_species_analysis(outputs, top_x=0)

    def test_plot_can_hide_legend(self):
        outputs = {
            "relative_abundances": [[0.1, 0.0, 0.02]],
            "species_ids": ["taxon-a"],
            "timeline_labels": ["Baseline", "ABX", "Day 20"],
            "follow_up_times": [20],
            "sequencing_depth": 1_000,
        }

        figure, axes = plot_returned_species_analysis(
            outputs,
            show_legend=False,
        )

        self.assertIsNone(axes.get_legend())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
