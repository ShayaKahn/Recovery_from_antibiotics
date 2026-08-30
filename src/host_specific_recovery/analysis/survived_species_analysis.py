from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import braycurtis, jensenshannon


@dataclass
class SurvivedSpeciesDOA:
    """Calculate survived-species dissimilarity-overlap analysis metrics.

    Parameters:
    baseline : array-like of shape (n_subjects, n_taxa)
        Baseline abundance table. Each row is one subject and each column is
        one taxon/species.
    abx : array-like of shape (n_subjects, n_taxa)
        Antibiotic-period abundance table. Rows and columns must be aligned
        with baseline.
    post_abx_cohorts : list of array-like, each of shape (n_subjects, n_taxa)
        Post-antibiotic abundance tables ordered by timepoint. Every table must
        have the same subject and taxon order as baseline.
    min_dissimilarity_species : int, default=10
        Minimum number of survived species required to calculate rJSD and
        Bray-Curtis dissimilarity. Subjects below this threshold receive NaN
        dissimilarity values.
    similarity_method : {"jaccard", "overlap"}, default="jaccard"
        Similarity metric to calculate for each ABX/post-ABX sample relative
        to baseline.
    dissimilarity_method : {"rjsd", "braycurtis"}, default="rjsd"
        Dissimilarity metric to calculate on the renormalized survived-species
        subspace.
    """

    baseline: np.ndarray
    abx: np.ndarray
    post_abx_cohorts: list[np.ndarray]
    min_dissimilarity_species: int = 10
    similarity_method: str = "jaccard"
    dissimilarity_method: str = "rjsd"

    def __post_init__(self):
        self.baseline = np.asarray(self.baseline, dtype=float)
        self.abx = np.asarray(self.abx, dtype=float)
        self.post_abx_cohorts = [
            np.asarray(post_abx_cohort, dtype=float)
            for post_abx_cohort in self.post_abx_cohorts
        ]
        self.similarity_method = self._normalize_similarity_method(self.similarity_method)
        self.dissimilarity_method = self._normalize_dissimilarity_method(self.dissimilarity_method)
        self._validate_inputs()

    @classmethod
    def from_dataset(
            cls,
            dataset: dict,
            min_dissimilarity_species: int = 10,
            similarity_method: str = "jaccard",
            dissimilarity_method: str = "rjsd",
    ) -> "SurvivedSpeciesDOA":
        return cls(
            baseline=dataset["baseline"],
            abx=dataset["abx"],
            post_abx_cohorts=dataset["post_abx_cohorts"],
            min_dissimilarity_species=min_dissimilarity_species,
            similarity_method=similarity_method,
            dissimilarity_method=dissimilarity_method,
        )

    @staticmethod
    def _normalize_similarity_method(method: str) -> str:
        method = method.lower()
        if method not in {"jaccard", "overlap"}:
            raise ValueError("similarity_method must be one of: 'jaccard', 'overlap'.")
        return method

    @staticmethod
    def _normalize_dissimilarity_method(method: str) -> str:
        method = method.lower()
        if method == "bc":
            method = "braycurtis"
        if method not in {"rjsd", "braycurtis"}:
            raise ValueError("dissimilarity_method must be one of: 'rjsd', 'braycurtis'.")
        return method

    def _validate_inputs(self):
        if self.baseline.ndim != 2:
            raise ValueError("baseline must be a 2D array with shape subjects x taxa.")
        if self.abx.shape != self.baseline.shape:
            raise ValueError("abx must have the same shape as baseline.")
        for cohort_idx, post_abx_cohort in enumerate(self.post_abx_cohorts):
            if post_abx_cohort.shape != self.baseline.shape:
                raise ValueError(
                    f"post_abx_cohorts[{cohort_idx}] must have the same shape as baseline."
                )

    @staticmethod
    def calc_overlap(first_sample: np.ndarray, second_sample: np.ndarray) -> float:
        shared_species = np.intersect1d(
            np.flatnonzero(first_sample),
            np.flatnonzero(second_sample),
        )
        return float(np.sum(first_sample[shared_species] + second_sample[shared_species]) / 2)

    @staticmethod
    def calc_jaccard(first_sample: np.ndarray, second_sample: np.ndarray) -> float:
        first_present = first_sample > 0
        second_present = second_sample > 0
        union = np.logical_or(first_present, second_present).sum()
        if union == 0:
            return np.nan
        intersection = np.logical_and(first_present, second_present).sum()
        return float(intersection / union)

    @staticmethod
    def normalize_subspace(sample: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        subspace_sample = sample[mask].astype(float)
        if subspace_sample.sum() == 0:
            return None
        return subspace_sample / subspace_sample.sum()

    @staticmethod
    def has_enough_species(mask: np.ndarray, min_species: int) -> bool:
        return np.count_nonzero(mask) >= min_species

    @staticmethod
    def calc_survived_species_mask(
            base: np.ndarray,
            abx_sample: np.ndarray,
            post_abx_samples: list[np.ndarray],
    ) -> np.ndarray:
        mask = (base > 0) & (abx_sample > 0)
        for post_abx_sample in post_abx_samples:
            mask &= post_abx_sample > 0
        return mask

    def calc_rjsd_to_baseline(
            self,
            base: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if not self.has_enough_species(survived_species_mask, self.min_dissimilarity_species):
            return np.nan

        base_subspace = self.normalize_subspace(base, survived_species_mask)
        sample_subspace = self.normalize_subspace(sample, survived_species_mask)
        if base_subspace is None or sample_subspace is None:
            return np.nan
        return float(jensenshannon(base_subspace, sample_subspace))

    def calc_bc_to_baseline(
            self,
            base: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if not self.has_enough_species(survived_species_mask, self.min_dissimilarity_species):
            return np.nan

        base_subspace = self.normalize_subspace(base, survived_species_mask)
        sample_subspace = self.normalize_subspace(sample, survived_species_mask)
        if base_subspace is None or sample_subspace is None:
            return np.nan
        return float(braycurtis(base_subspace, sample_subspace))

    def calc_similarity(self, base: np.ndarray, sample: np.ndarray) -> float:
        if self.similarity_method == "jaccard":
            return self.calc_jaccard(base, sample)
        if self.similarity_method == "overlap":
            return self.calc_overlap(base, sample)
        raise ValueError("similarity_method must be one of: 'jaccard', 'overlap'.")

    def calc_dissimilarity(
            self,
            base: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if self.dissimilarity_method == "rjsd":
            return self.calc_rjsd_to_baseline(base, sample, survived_species_mask)
        if self.dissimilarity_method == "braycurtis":
            return self.calc_bc_to_baseline(base, sample, survived_species_mask)
        raise ValueError("dissimilarity_method must be one of: 'rjsd', 'braycurtis'.")

    def calculate(self) -> dict:
        similarity_values = []
        dissimilarity_values = []
        n_survived_species = []

        for subject_idx, base in enumerate(self.baseline):
            abx_sample = self.abx[subject_idx]
            post_abx_samples = [
                post_abx_cohort[subject_idx]
                for post_abx_cohort in self.post_abx_cohorts
            ]
            survived_species_mask = self.calc_survived_species_mask(
                base,
                abx_sample,
                post_abx_samples,
            )

            subject_samples = [abx_sample, *post_abx_samples]
            similarity_values.append([
                self.calc_similarity(base, sample)
                for sample in subject_samples
            ])
            dissimilarity_values.append([
                self.calc_dissimilarity(base, sample, survived_species_mask)
                for sample in subject_samples
            ])
            n_survived_species.append(np.count_nonzero(survived_species_mask))

        similarity_values = np.asarray(similarity_values, dtype=float)
        dissimilarity_values = np.asarray(dissimilarity_values, dtype=float)

        return {
            "similarity_values": similarity_values,
            "dissimilarity_values": dissimilarity_values,
            "n_survived_species": np.asarray(n_survived_species, dtype=int),
        }


@dataclass
class DenseTimeSeriesSurvivedSpeciesDOA:
    """Calculate survived-species DOA metrics for one dense time-series subject.

    Parameters:
    samples : array-like of shape (n_samples, n_taxa)
        Abundance table for one subject. Rows are samples/timepoints ordered in
        time, and columns are taxa/species.
    baseline_indexes : range, slice, or array-like of int
        Row indexes in samples that correspond to baseline samples. All
        non-baseline rows are compared to these baseline rows.
    min_dissimilarity_species : int, default=10
        Minimum number of survived species required to calculate rJSD and
        Bray-Curtis dissimilarity. If fewer survived species are present, the
        dissimilarity outputs are NaN.
    similarity_method : {"jaccard", "overlap"}, default="jaccard"
        Similarity metric to calculate for each non-baseline sample relative
        to each baseline sample.
    dissimilarity_method : {"rjsd", "braycurtis"}, default="rjsd"
        Dissimilarity metric to calculate on the renormalized survived-species
        subspace.
    survived_species_min_prevalence : float, default=1.0
        Minimum fraction of samples where a taxon must be present to be defined
        as a survived species. The default 1.0 requires presence in every
        sample.

    Notes:
    Similarity is calculated on the full taxa vector, but only for
    baseline/sample pairs where dissimilarity can also be calculated. A pair is
    valid only when both samples contain every selected survived species.
    """

    samples: np.ndarray
    baseline_indexes: range | slice | list[int] | np.ndarray
    min_dissimilarity_species: int = 10
    similarity_method: str = "jaccard"
    dissimilarity_method: str = "rjsd"
    survived_species_min_prevalence: float = 1.0

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        self.baseline_indexes = self._normalize_baseline_indexes(self.baseline_indexes)
        self.similarity_method = SurvivedSpeciesDOA._normalize_similarity_method(self.similarity_method)
        self.dissimilarity_method = SurvivedSpeciesDOA._normalize_dissimilarity_method(self.dissimilarity_method)
        self._validate_inputs()

    def _normalize_baseline_indexes(self, baseline_indexes) -> np.ndarray:
        if isinstance(baseline_indexes, slice):
            if self.samples.ndim != 2:
                raise ValueError("samples must be a 2D array before slice indexes can be resolved.")
            return np.arange(self.samples.shape[0])[baseline_indexes]

        return np.asarray(list(baseline_indexes), dtype=int)

    def _validate_inputs(self):
        if self.samples.ndim != 2:
            raise ValueError("samples must be a 2D array with shape samples x taxa.")
        if self.samples.shape[0] == 0:
            raise ValueError("samples must contain at least one sample.")
        if self.samples.shape[1] == 0:
            raise ValueError("samples must contain at least one taxon.")
        if self.baseline_indexes.ndim != 1 or self.baseline_indexes.size == 0:
            raise ValueError("baseline_indexes must contain at least one row index.")
        if np.unique(self.baseline_indexes).size != self.baseline_indexes.size:
            raise ValueError("baseline_indexes must not contain duplicate indexes.")
        if self.baseline_indexes.min() < 0 or self.baseline_indexes.max() >= self.samples.shape[0]:
            raise ValueError("baseline_indexes contains indexes outside the sample matrix.")
        if self.baseline_indexes.size == self.samples.shape[0]:
            raise ValueError("At least one non-baseline sample is required.")
        if (
                not isinstance(self.survived_species_min_prevalence, (int, float))
                or not (0 < self.survived_species_min_prevalence <= 1)
        ):
            raise ValueError("survived_species_min_prevalence must be a number in the interval (0, 1].")

    def calc_survived_species_mask(self) -> np.ndarray:
        prevalence = np.mean(self.samples > 0, axis=0)
        return prevalence >= self.survived_species_min_prevalence

    @staticmethod
    def sample_contains_survived_species(sample: np.ndarray, survived_species_mask: np.ndarray) -> bool:
        return bool(np.all(sample[survived_species_mask] > 0))

    def calc_rjsd(
            self,
            baseline_sample: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if not SurvivedSpeciesDOA.has_enough_species(
                survived_species_mask,
                self.min_dissimilarity_species,
        ):
            return np.nan

        baseline_subspace = SurvivedSpeciesDOA.normalize_subspace(
            baseline_sample,
            survived_species_mask,
        )
        sample_subspace = SurvivedSpeciesDOA.normalize_subspace(
            sample,
            survived_species_mask,
        )
        if baseline_subspace is None or sample_subspace is None:
            return np.nan
        return float(jensenshannon(baseline_subspace, sample_subspace))

    def calc_braycurtis(
            self,
            baseline_sample: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if not SurvivedSpeciesDOA.has_enough_species(
                survived_species_mask,
                self.min_dissimilarity_species,
        ):
            return np.nan

        baseline_subspace = SurvivedSpeciesDOA.normalize_subspace(
            baseline_sample,
            survived_species_mask,
        )
        sample_subspace = SurvivedSpeciesDOA.normalize_subspace(
            sample,
            survived_species_mask,
        )
        if baseline_subspace is None or sample_subspace is None:
            return np.nan
        return float(braycurtis(baseline_subspace, sample_subspace))

    def calc_similarity(self, baseline_sample: np.ndarray, sample: np.ndarray) -> float:
        if self.similarity_method == "jaccard":
            return SurvivedSpeciesDOA.calc_jaccard(baseline_sample, sample)
        if self.similarity_method == "overlap":
            return SurvivedSpeciesDOA.calc_overlap(baseline_sample, sample)
        raise ValueError("similarity_method must be one of: 'jaccard', 'overlap'.")

    def calc_dissimilarity(
            self,
            baseline_sample: np.ndarray,
            sample: np.ndarray,
            survived_species_mask: np.ndarray,
    ) -> float:
        if self.dissimilarity_method == "rjsd":
            return self.calc_rjsd(baseline_sample, sample, survived_species_mask)
        if self.dissimilarity_method == "braycurtis":
            return self.calc_braycurtis(baseline_sample, sample, survived_species_mask)
        raise ValueError("dissimilarity_method must be one of: 'rjsd', 'braycurtis'.")

    def _average_to_baselines(self, sample: np.ndarray, survived_species_mask: np.ndarray) -> dict:
        baseline_samples = self.samples[self.baseline_indexes]

        similarity = []
        dissimilarity = []
        has_enough_species = SurvivedSpeciesDOA.has_enough_species(
            survived_species_mask,
            self.min_dissimilarity_species,
        )
        sample_is_valid = self.sample_contains_survived_species(sample, survived_species_mask)

        for baseline_sample in baseline_samples:
            baseline_is_valid = self.sample_contains_survived_species(
                baseline_sample,
                survived_species_mask,
            )
            if not has_enough_species or not sample_is_valid or not baseline_is_valid:
                similarity.append(np.nan)
                dissimilarity.append(np.nan)
                continue

            similarity.append(self.calc_similarity(baseline_sample, sample))
            dissimilarity.append(
                self.calc_dissimilarity(
                    baseline_sample,
                    sample,
                    survived_species_mask,
                )
            )

        return {
            "similarity": np.asarray(similarity, dtype=float),
            "dissimilarity": np.asarray(dissimilarity, dtype=float),
        }

    def calculate(self) -> dict:
        baseline_index_set = set(self.baseline_indexes.tolist())
        sample_indices = np.asarray([
            sample_idx
            for sample_idx in range(self.samples.shape[0])
            if sample_idx not in baseline_index_set
        ], dtype=int)
        survived_species_mask = self.calc_survived_species_mask()

        similarity_to_baseline = []
        dissimilarity_to_baseline = []

        for sample_idx in sample_indices:
            pairwise_values = self._average_to_baselines(
                self.samples[sample_idx],
                survived_species_mask,
            )
            similarity_to_baseline.append(pairwise_values["similarity"])
            dissimilarity_to_baseline.append(pairwise_values["dissimilarity"])

        similarity_to_baseline = np.asarray(similarity_to_baseline, dtype=float)
        dissimilarity_to_baseline = np.asarray(dissimilarity_to_baseline, dtype=float)
        valid_similarity = np.isfinite(similarity_to_baseline)
        valid_dissimilarity = np.isfinite(dissimilarity_to_baseline)
        similarity_values = np.asarray([
            np.mean(row[valid_row]) if np.any(valid_row) else np.nan
            for row, valid_row in zip(similarity_to_baseline, valid_similarity)
        ])
        dissimilarity_values = np.asarray([
            np.mean(row[valid_row]) if np.any(valid_row) else np.nan
            for row, valid_row in zip(dissimilarity_to_baseline, valid_dissimilarity)
        ])

        return {
            "similarity_values": similarity_values,
            "dissimilarity_values": dissimilarity_values,
            "n_survived_species": int(np.count_nonzero(survived_species_mask)),
        }


def run_survived_species_doa_analysis(
        dataset: dict,
        min_dissimilarity_species: int = 10,
        similarity_method: str = "jaccard",
        dissimilarity_method: str = "rjsd",
) -> dict:
    analysis = SurvivedSpeciesDOA.from_dataset(
        dataset,
        min_dissimilarity_species=min_dissimilarity_species,
        similarity_method=similarity_method,
        dissimilarity_method=dissimilarity_method,
    )
    return analysis.calculate()


def run_survived_species_analysis(dataset: dict, scale=1e2) -> dict:

    baseline = dataset["baseline"]
    ABX = dataset["abx"]
    posts = [np.asarray(P, float) for P in dataset["post_abx_cohorts"]]

    m, _ = baseline.shape
    results = []

    for i in range(m):
        base = baseline[i, :]
        abx = ABX[i, :]
        post_rows = [P[i, :] for P in posts]

        mask = (base > 0) & (abx > 0)
        for p in post_rows:
            mask &= (p > 0)

        ratio = np.zeros_like(base, dtype=float)
        np.divide(base, abx, out=ratio, where=abx != 0)
        mask &= ratio > scale

        s = np.where(mask)[0]
        if s.size:
            measurements = [base[s], abx[s], *[p[s] for p in post_rows]]
            for j in range(np.size(measurements[0])):
                results.append(np.log10([val[j] for val in measurements]))

    results_matrix = np.vstack(results)
    mean = np.mean(results_matrix, axis=0)

    return {
        "results_matrix": results_matrix,
        "mean": mean
    }


def run_dense_time_series_survived_species_analysis(
        samples,
        baseline_indexes,
        abx_indexes,
        scale=1e2,
        survived_species_min_prevalence=1.0,
) -> dict:
    """Analyze reduced survived species for one dense time-series subject.

    Parameters:
    samples : array-like of shape (n_samples, n_taxa)
        Abundance table for one subject. Rows are samples/timepoints and
        columns are taxa/species.
    baseline_indexes : range, slice, or array-like of int
        Row indexes corresponding to baseline samples.
    abx_indexes : range, slice, or array-like of int
        Row indexes corresponding to antibiotic-period samples.
    scale : float, default=1e2
        Minimum fold-reduction threshold. A survived species is retained if
        mean baseline abundance / minimum ABX abundance is greater than scale.
    survived_species_min_prevalence : float, default=1.0
        Minimum fraction of samples where a taxon must be present to be defined
        as survived. In addition, survived species must be present in every ABX
        sample. The default 1.0 requires presence in every sample.

    Returns:
    dict
        results_matrix contains log10 abundances for retained species across
        all samples. Zero abundance entries are represented as np.nan. plot_matrix
        replaces those np.nan entries with log10(abundance_floor), where
        abundance_floor is half the minimum positive abundance among retained
        species.
    """

    samples = np.asarray(samples, dtype=float)
    baseline_indexes = _normalize_dense_time_series_indexes(samples, baseline_indexes, "baseline_indexes")
    abx_indexes = _normalize_dense_time_series_indexes(samples, abx_indexes, "abx_indexes")

    if (
            not isinstance(survived_species_min_prevalence, (int, float))
            or not (0 < survived_species_min_prevalence <= 1)
    ):
        raise ValueError("survived_species_min_prevalence must be a number in the interval (0, 1].")
    if not isinstance(scale, (int, float)) or scale < 0:
        raise ValueError("scale must be a non-negative number.")

    prevalence = np.mean(samples > 0, axis=0)
    present_in_all_abx = np.all(samples[abx_indexes] > 0, axis=0)
    survived_species_mask = (prevalence >= survived_species_min_prevalence) & present_in_all_abx

    baseline_mean = samples[baseline_indexes].mean(axis=0)
    abx_min = samples[abx_indexes].min(axis=0)

    ratio = np.zeros_like(baseline_mean, dtype=float)
    np.divide(baseline_mean, abx_min, out=ratio, where=abx_min != 0)

    reduced_species_mask = survived_species_mask & (ratio > scale)
    reduced_species_indexes = np.flatnonzero(reduced_species_mask)

    if reduced_species_indexes.size == 0:
        results_matrix = np.empty((0, samples.shape[0]), dtype=float)
        plot_matrix = results_matrix.copy()
        mean = np.full(samples.shape[0], np.nan, dtype=float)
        plot_mean = mean.copy()
        abundance_floor = np.nan
    else:
        selected_abundances = samples[:, reduced_species_indexes].T
        results_matrix = np.full(selected_abundances.shape, np.nan, dtype=float)
        positive_mask = selected_abundances > 0
        results_matrix[positive_mask] = np.log10(selected_abundances[positive_mask])
        valid_counts = np.sum(np.isfinite(results_matrix), axis=0)
        column_sums = np.nansum(results_matrix, axis=0)
        mean = np.divide(
            column_sums,
            valid_counts,
            out=np.full(samples.shape[0], np.nan, dtype=float),
            where=valid_counts > 0,
        )
        abundance_floor = selected_abundances[positive_mask].min() / 2
        plot_matrix = np.full(selected_abundances.shape, np.log10(abundance_floor), dtype=float)
        plot_matrix[positive_mask] = results_matrix[positive_mask]
        plot_mean = plot_matrix.mean(axis=0)

    return {
        "results_matrix": results_matrix,
        "plot_matrix": plot_matrix,
        "mean": mean,
        "plot_mean": plot_mean,
        "abundance_floor": abundance_floor,
        "n_survived_species": int(np.count_nonzero(survived_species_mask)),
        "n_reduced_species": int(reduced_species_indexes.size),
        "reduced_species_indexes": reduced_species_indexes,
    }


def _normalize_dense_time_series_indexes(samples: np.ndarray, indexes, name: str) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array with shape samples x taxa.")

    if isinstance(indexes, slice):
        indexes = np.arange(samples.shape[0])[indexes]
    else:
        indexes = np.asarray(list(indexes), dtype=int)

    if indexes.ndim != 1 or indexes.size == 0:
        raise ValueError(f"{name} must contain at least one row index.")
    if np.unique(indexes).size != indexes.size:
        raise ValueError(f"{name} must not contain duplicate indexes.")
    if indexes.min() < 0 or indexes.max() >= samples.shape[0]:
        raise ValueError(f"{name} contains indexes outside the sample matrix.")

    return indexes
