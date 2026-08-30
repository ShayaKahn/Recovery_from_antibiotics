from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class ReturnedSpeciesAnalysis:
    """Find taxa whose first post-antibiotic detection is the final follow-up."""

    __slots__ = ("baseline", "abx", "post_abx_cohorts", "species_ids", "subject_ids", "follow_up_times",
                 "sequencing_depth", "sample_depth", "sample_depth_timepoints", "non_rarefied",
                 "survived_reduction_threshold")

    SUMMARY_COLUMNS = ["species_id", "species_index", "subject_id", "baseline_relative_abundance",
                       "last_follow_up_relative_abundance"]
    REDUCED_SURVIVED_SUMMARY_COLUMNS = [
        "species_id",
        "species_index",
        "subject_id",
        "baseline_relative_abundance",
        "abx_relative_abundance",
        "baseline_to_abx_fold_reduction",
    ]
    FIRST_RETURNED_SUMMARY_COLUMNS = [
        "species_id",
        "species_index",
        "subject_id",
        "baseline_relative_abundance",
        "first_post_abx_relative_abundance",
    ]

    def __init__(self, dataset: Mapping, species_ids: Sequence | None = None, subject_ids: Sequence | None = None,
                 follow_up_times: Sequence | None = None, sequencing_depth: float | None = None,
                 non_rarefied: bool = False, sample_depth_timepoints: Sequence[str] | None = None,
                 survived_reduction_threshold: float = 1e2):
        """Initialize the analysis from a dataset.

        By default, the dataset must contain ``baseline``, ``abx``, and
        ``post_abx_cohorts``. When ``non_rarefied=True``, their ``_no_rar``
        counterparts are used instead. The ``abx`` matrix represents the
        final sample collected during treatment. All matrices must contain
        relative abundances and have shape ``(n_subjects, n_species)``.

        ``sequencing_depth`` is the common rarefaction depth ``N_0``. When it
        is not supplied explicitly, it is read from ``dataset["depth"]``.
        Non-rarefied analyses instead read sample-specific depths from the
        subject-by-timepoint ``dataset["sample_depth"]`` table.
        """
        if not isinstance(dataset, Mapping):
            raise TypeError("dataset must be a mapping.")
        if not isinstance(non_rarefied, (bool, np.bool_)):
            raise TypeError("non_rarefied must be a boolean.")
        if (
                not isinstance(survived_reduction_threshold, (int, float))
                or not np.isfinite(survived_reduction_threshold)
                or survived_reduction_threshold <= 0
        ):
            raise ValueError("survived_reduction_threshold must be a positive finite number.")

        self.non_rarefied = bool(non_rarefied)
        self.survived_reduction_threshold = float(survived_reduction_threshold)
        suffix = "_no_rar" if self.non_rarefied else ""
        baseline_key = f"baseline{suffix}"
        abx_key = f"abx{suffix}"
        post_abx_cohorts_key = f"post_abx_cohorts{suffix}"

        required_keys = {baseline_key, abx_key, post_abx_cohorts_key}
        missing_keys = required_keys.difference(dataset)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"dataset is missing required field(s): {missing}.")

        self.baseline = self._as_abundance_matrix(dataset[baseline_key], baseline_key)
        self.abx = self._as_abundance_matrix(dataset[abx_key], abx_key)
        self.post_abx_cohorts = [
            self._as_abundance_matrix(
                cohort,
                f"{post_abx_cohorts_key}[{index}]",
            )
            for index, cohort in enumerate(dataset[post_abx_cohorts_key])
        ]

        if not self.post_abx_cohorts:
            raise ValueError("post_abx_cohorts must contain at least one follow-up matrix.")
        if self.abx.shape != self.baseline.shape:
            raise ValueError("abx must have the same shape as baseline.")
        for index, cohort in enumerate(self.post_abx_cohorts):
            if cohort.shape != self.baseline.shape:
                raise ValueError(f"post_abx_cohorts[{index}] must have the same shape as baseline.")

        n_subjects, n_species = self.baseline.shape
        inferred_species_ids = self._infer_species_ids(
            dataset,
            n_species,
            non_rarefied=self.non_rarefied,
        )
        inferred_subject_ids = self._infer_subject_ids(dataset, n_subjects)
        inferred_follow_up_times = dataset.get("times_post_abx", np.arange(len(self.post_abx_cohorts)))

        self.species_ids = self._validate_labels(inferred_species_ids if species_ids is None else species_ids,
                                                 n_species,"species_ids")
        self.subject_ids = self._validate_labels(inferred_subject_ids if subject_ids is None else subject_ids,
                                                 n_subjects,"subject_ids", require_unique=True)
        self.follow_up_times = self._validate_labels(
            inferred_follow_up_times if follow_up_times is None else follow_up_times,
            len(self.post_abx_cohorts),"follow_up_times")

        if self.non_rarefied:
            if sequencing_depth is not None:
                raise ValueError(
                    "sequencing_depth cannot be supplied when non_rarefied=True; "
                    "sample-specific depths are read from dataset['sample_depth']."
                )
            self.sample_depth, self.sample_depth_timepoints = self._validate_sample_depths(
                dataset,
                self.subject_ids,
                len(self.post_abx_cohorts),
                sample_depth_timepoints,
            )
            self.sequencing_depth = None
        else:
            if sequencing_depth is None:
                if "depth" not in dataset:
                    raise KeyError("dataset is missing required field 'depth'; alternatively,"
                                   " pass sequencing_depth explicitly.")
                sequencing_depth = dataset["depth"]
            self.sequencing_depth = self._validate_sequencing_depth(sequencing_depth)
            self.sample_depth = None
            self.sample_depth_timepoints = None

    @staticmethod
    def _as_abundance_matrix(values, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a 2D subjects x species matrix.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain only finite abundances.")
        if np.any(matrix < 0):
            raise ValueError(f"{name} must contain non-negative abundances.")
        return matrix

    @staticmethod
    def _validate_labels(values, expected_size: int, name: str, require_unique: bool = False) -> np.ndarray:
        labels = np.asarray(list(values), dtype=object)
        if labels.ndim != 1 or labels.size != expected_size:
            raise ValueError(f"{name} must contain exactly {expected_size} values.")
        if require_unique and pd.Index(labels).has_duplicates:
            raise ValueError(f"{name} must not contain duplicate values.")
        return labels

    @staticmethod
    def _validate_sequencing_depth(value) -> float:
        try:
            depth = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("sequencing_depth must be a positive finite number.") from error
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("sequencing_depth must be a positive finite number.")
        return depth

    @staticmethod
    def _validate_sample_depths(
            dataset: Mapping,
            subject_ids: np.ndarray,
            n_follow_ups: int,
            sample_depth_timepoints: Sequence[str] | None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if "sample_depth" not in dataset:
            raise KeyError(
                "dataset is missing required field 'sample_depth' for a non-rarefied analysis."
            )

        sample_depth = dataset["sample_depth"]
        if not isinstance(sample_depth, pd.DataFrame):
            raise TypeError(
                "dataset['sample_depth'] must be a subject-by-timepoint DataFrame."
            )
        if sample_depth.index.has_duplicates or sample_depth.columns.has_duplicates:
            raise ValueError("dataset['sample_depth'] must have unique index and column labels.")

        missing_subjects = [
            subject_id for subject_id in subject_ids
            if subject_id not in sample_depth.index
        ]
        if missing_subjects:
            missing = ", ".join(map(str, missing_subjects))
            raise KeyError(f"sample_depth is missing required subject(s): {missing}.")

        if sample_depth_timepoints is None:
            post_abx_timepoints = [
                column for column in sample_depth.columns
                if str(column).startswith("post_abx")
            ]
            sample_depth_timepoints = ["baseline", "abx", *post_abx_timepoints]
        else:
            sample_depth_timepoints = list(sample_depth_timepoints)

        expected_timepoints = n_follow_ups + 2
        if len(sample_depth_timepoints) != expected_timepoints:
            raise ValueError(
                "sample_depth_timepoints must contain Baseline, ABX, and exactly "
                f"{n_follow_ups} post-ABX timepoints."
            )
        if len(set(sample_depth_timepoints)) != len(sample_depth_timepoints):
            raise ValueError("sample_depth_timepoints must contain unique labels.")

        missing_timepoints = [
            timepoint for timepoint in sample_depth_timepoints
            if timepoint not in sample_depth.columns
        ]
        if missing_timepoints:
            missing = ", ".join(map(str, missing_timepoints))
            raise KeyError(f"sample_depth is missing required timepoint(s): {missing}.")

        aligned_depths = sample_depth.loc[subject_ids, sample_depth_timepoints].astype(float)
        values = aligned_depths.to_numpy()
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("All sample-specific sequencing depths must be positive and finite.")

        return aligned_depths, np.asarray(sample_depth_timepoints, dtype=object)

    @staticmethod
    def _infer_species_ids(dataset: Mapping, n_species: int, non_rarefied: bool = False):
        if "taxa_ids" in dataset and len(dataset["taxa_ids"]) == n_species:
            return dataset["taxa_ids"]

        if non_rarefied:
            baseline_df = dataset.get(
                "baseline_df_no_rar",
                dataset.get("df_baseline_no_rar"),
            )
        else:
            baseline_df = dataset.get("baseline_df", dataset.get("df_baseline"))
        if isinstance(baseline_df, pd.DataFrame) and len(baseline_df.index) == n_species:
            return baseline_df.index

        return np.arange(n_species)

    @staticmethod
    def _infer_subject_ids(dataset: Mapping, n_subjects: int):
        for key in ("filtered_keys", "subject_ids", "keys"):
            if key in dataset and len(dataset[key]) == n_subjects:
                return dataset[key]
        return np.arange(n_subjects)

    def _subject_index(self, subject_id) -> int:
        matches = np.flatnonzero(self.subject_ids == subject_id)
        if matches.size == 0:
            available = ", ".join(map(str, self.subject_ids))
            raise KeyError(f"Unknown subject_id {subject_id!r}. Available subject IDs: {available}.")
        return int(matches[0])

    def run(self, subject_id) -> dict:
        """Analyze taxa that first reappear in a subject's final follow-up.

        A returned taxon must be detected at baseline, undetected in the final
        ABX sample and every earlier post-ABX sample, and detected in the last
        post-ABX sample.

        Returns a dictionary containing a species summary and the complete
        relative-abundance trajectory for each selected taxon. The trajectory
        columns are ordered as Baseline, ABX, and all post-ABX samples.
        """
        subject_index = self._subject_index(subject_id)
        baseline_sample = self.baseline[subject_index]
        abx_sample = self.abx[subject_index]
        subject_follow_ups = np.stack([cohort[subject_index] for cohort in self.post_abx_cohorts], axis=0)

        returned_mask = ((baseline_sample > 0) & (abx_sample == 0) & np.all(subject_follow_ups[:-1] == 0, axis=0)
                         & (subject_follow_ups[-1] > 0))
        species_indices = np.flatnonzero(returned_mask)
        returned_species_ids = self.species_ids[species_indices]

        first_returned_mask = (
            (baseline_sample > 0)
            & (abx_sample == 0)
            & np.all(subject_follow_ups > 0, axis=0)
        )
        first_returned_species_indices = np.flatnonzero(first_returned_mask)
        first_returned_species_ids = self.species_ids[
            first_returned_species_indices
        ]

        survived_mask = (
            (baseline_sample > 0)
            & (abx_sample > 0)
            & np.all(subject_follow_ups > 0, axis=0)
        )
        baseline_to_abx_fold_reduction = np.zeros_like(baseline_sample, dtype=float)
        np.divide(
            baseline_sample,
            abx_sample,
            out=baseline_to_abx_fold_reduction,
            where=abx_sample > 0,
        )
        reduced_survived_mask = (
            survived_mask
            & (baseline_to_abx_fold_reduction > self.survived_reduction_threshold)
        )
        reduced_survived_species_indices = np.flatnonzero(reduced_survived_mask)
        reduced_survived_species_ids = self.species_ids[
            reduced_survived_species_indices
        ]

        all_samples = np.vstack([baseline_sample, abx_sample, subject_follow_ups])
        relative_abundances = all_samples[:, species_indices].T
        baseline_abundances = baseline_sample[species_indices]
        last_follow_up_abundances = subject_follow_ups[-1, species_indices]
        reduced_survived_relative_abundances = all_samples[
            :,
            reduced_survived_species_indices,
        ].T
        reduced_survived_fold_reductions = baseline_to_abx_fold_reduction[
            reduced_survived_species_indices
        ]
        first_returned_relative_abundances = all_samples[
            :,
            first_returned_species_indices,
        ].T

        if self.non_rarefied:
            sequencing_depths = self.sample_depth.loc[subject_id].to_numpy(dtype=float)
            sequencing_depth = sequencing_depths.copy()
        else:
            sequencing_depths = np.full(all_samples.shape[0], self.sequencing_depth)
            sequencing_depth = self.sequencing_depth
        one_read_abundances = 1.0 / sequencing_depths
        detection_thresholds = 3.0 / sequencing_depths

        summary = pd.DataFrame({
            "species_id": returned_species_ids,
            "species_index": species_indices.astype(int),
            "subject_id": np.repeat(subject_id, species_indices.size),
            "baseline_relative_abundance": baseline_abundances,
            "last_follow_up_relative_abundance": last_follow_up_abundances,
        }, columns=self.SUMMARY_COLUMNS)
        reduced_survived_summary = pd.DataFrame({
            "species_id": reduced_survived_species_ids,
            "species_index": reduced_survived_species_indices.astype(int),
            "subject_id": np.repeat(
                subject_id,
                reduced_survived_species_indices.size,
            ),
            "baseline_relative_abundance": baseline_sample[
                reduced_survived_species_indices
            ],
            "abx_relative_abundance": abx_sample[
                reduced_survived_species_indices
            ],
            "baseline_to_abx_fold_reduction": reduced_survived_fold_reductions,
        }, columns=self.REDUCED_SURVIVED_SUMMARY_COLUMNS)
        first_returned_summary = pd.DataFrame({
            "species_id": first_returned_species_ids,
            "species_index": first_returned_species_indices.astype(int),
            "subject_id": np.repeat(
                subject_id,
                first_returned_species_indices.size,
            ),
            "baseline_relative_abundance": baseline_sample[
                first_returned_species_indices
            ],
            "first_post_abx_relative_abundance": subject_follow_ups[0][
                first_returned_species_indices
            ],
        }, columns=self.FIRST_RETURNED_SUMMARY_COLUMNS)

        return {
            "subject_id": subject_id,
            "subject_index": subject_index,
            "returned_species": summary,
            "species_ids": returned_species_ids.copy(),
            "species_indices": species_indices,
            "baseline_relative_abundance": baseline_abundances,
            "last_follow_up_relative_abundance": last_follow_up_abundances,
            "relative_abundances": relative_abundances,
            "detected": relative_abundances > 0,
            "first_returned_species": first_returned_summary,
            "first_returned_species_ids": first_returned_species_ids.copy(),
            "first_returned_species_indices": first_returned_species_indices,
            "first_returned_relative_abundances": first_returned_relative_abundances,
            "reduced_survived_species": reduced_survived_summary,
            "reduced_survived_species_ids": reduced_survived_species_ids.copy(),
            "reduced_survived_species_indices": reduced_survived_species_indices,
            "reduced_survived_relative_abundances": reduced_survived_relative_abundances,
            "survived_fold_reductions": reduced_survived_fold_reductions,
            "survived_reduction_threshold": self.survived_reduction_threshold,
            "timeline_labels": np.asarray([
                "Baseline",
                "ABX",
                *[f"Day {time}" for time in self.follow_up_times],
            ], dtype=object),
            "follow_up_times": self.follow_up_times.copy(),
            "last_follow_up_time": self.follow_up_times[-1],
            "sequencing_depth": sequencing_depth,
            "sequencing_depths": sequencing_depths,
            "sample_depth_timepoints": (
                self.sample_depth_timepoints.copy()
                if self.sample_depth_timepoints is not None
                else None
            ),
            "one_read_abundances": one_read_abundances,
            "detection_threshold": (
                detection_thresholds
                if self.non_rarefied
                else float(detection_thresholds[0])
            ),
            "detection_thresholds": detection_thresholds,
            "non_rarefied": self.non_rarefied,
            "n_returned_species": int(species_indices.size),
            "n_first_returned_species": int(first_returned_species_indices.size),
            "n_reduced_survived_species": int(reduced_survived_species_indices.size),
        }
