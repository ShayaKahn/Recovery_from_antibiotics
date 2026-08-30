"""Reference-cohort size sensitivity analysis for surrogate data analysis."""

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.host_specific_recovery.analysis.surrogate_data_analysis import (
    calculate_sda_statistics,
)
from src.host_specific_recovery.utils.surrogate_data_analysis_utils import (
    create_cohort_dict,
)


def _validate_dataset(dataset: Mapping):
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping.")

    required_fields = {
        "baseline",
        "baseline_full",
        "abx",
        "post_abx_cohorts",
        "keys",
        "filtered_keys",
    }
    missing_fields = required_fields.difference(dataset)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise KeyError(f"dataset is missing required field(s): {missing}.")

    baseline = np.asarray(dataset["baseline"], dtype=float)
    baseline_full = np.asarray(dataset["baseline_full"], dtype=float)
    abx = np.asarray(dataset["abx"], dtype=float)
    post_abx_cohorts = [
        np.asarray(cohort, dtype=float)
        for cohort in dataset["post_abx_cohorts"]
    ]
    keys = np.asarray(list(dataset["keys"]), dtype=object)
    subject_ids = np.asarray(list(dataset["filtered_keys"]), dtype=object)

    if baseline.ndim != 2 or baseline_full.ndim != 2 or abx.ndim != 2:
        raise ValueError("baseline, baseline_full, and abx must be 2D matrices.")
    if baseline.shape != abx.shape:
        raise ValueError("baseline and abx must have the same shape.")
    if baseline.shape[0] != subject_ids.size:
        raise ValueError("filtered_keys must contain one ID per baseline subject.")
    if baseline_full.shape[0] != keys.size:
        raise ValueError("keys must contain one ID per baseline_full row.")
    if baseline_full.shape[1] != baseline.shape[1]:
        raise ValueError("baseline_full and baseline must contain the same species.")
    if not post_abx_cohorts:
        raise ValueError("post_abx_cohorts must contain at least one follow-up matrix.")
    if any(cohort.shape != baseline.shape for cohort in post_abx_cohorts):
        raise ValueError(
            "Every post_abx_cohorts matrix must have the same shape as baseline."
        )

    abundance_matrices = [baseline, baseline_full, abx, *post_abx_cohorts]
    if any(not np.all(np.isfinite(matrix)) for matrix in abundance_matrices):
        raise ValueError("Abundance matrices must contain only finite values.")
    if any(np.any(matrix < 0) for matrix in abundance_matrices):
        raise ValueError("Abundance matrices must contain non-negative values.")

    return baseline, baseline_full, abx, post_abx_cohorts, keys, subject_ids


def _validate_subset_sizes(subset_sizes: Sequence[int], max_size: int) -> list[int]:
    if isinstance(subset_sizes, (str, bytes)):
        raise TypeError("subset_sizes must be a sequence of integers.")
    try:
        subset_sizes = list(subset_sizes)
    except TypeError as exc:
        raise TypeError("subset_sizes must be a sequence of integers.") from exc
    if not subset_sizes:
        raise ValueError("subset_sizes must contain at least one size.")
    if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            for size in subset_sizes
    ):
        raise TypeError("Every subset size must be an integer.")

    normalized_sizes = sorted(set(int(size) for size in subset_sizes))
    if normalized_sizes[0] < 2:
        raise ValueError(
            "Every subset size must be at least 2 so standardized similarity can be estimated."
        )
    if normalized_sizes[-1] > max_size:
        raise ValueError(
            f"subset_sizes cannot exceed the smallest reference pool ({max_size})."
        )
    return normalized_sizes


def _calculate_subject_similarity_profiles(
        baseline,
        baseline_full,
        abx,
        post_abx_cohorts,
        keys,
        subject_ids,
        timepoints_val,
        method,
        strict,
        naive,
):
    from src.host_specific_recovery.statistical_models.surrogate import Surrogate

    profiles = []
    for subject_index, subject_id in enumerate(subject_ids):
        subject_key = str(subject_id)
        reference_baselines = create_cohort_dict(
            keys,
            subject_id,
            baseline_full,
        )
        if not reference_baselines:
            raise ValueError(
                f"No eligible reference baselines remain for subject {subject_id!r}."
            )

        post_abx_matrix = np.vstack([
            cohort[subject_index]
            for cohort in post_abx_cohorts
        ])
        surrogate = Surrogate(
            reference_baselines,
            {subject_key: baseline[subject_index]},
            post_abx_matrix,
            abx[subject_index],
            timepoints=timepoints_val,
            strict=strict,
            naive=naive,
        )
        similarities = surrogate.apply_surrogate_data_analysis(method=method)
        reference_ids = np.asarray(list(reference_baselines), dtype=object)
        profiles.append({
            "subject_id": subject_id,
            "observed_similarity": float(similarities[subject_key]),
            "reference_ids": reference_ids,
            "reference_similarities": np.asarray(
                [similarities[reference_id] for reference_id in reference_ids],
                dtype=float,
            ),
        })
    return profiles


def run_reference_subsampling_analysis(
        dataset: Mapping,
        subset_sizes: Sequence[int],
        n_repeats: int = 1000,
        timepoints_val: int = 0,
        method: str = "Jaccard",
        alpha: float = 0.05,
        random_state: int | None = 0,
        strict: bool = False,
        naive: bool = False,
) -> pd.DataFrame:
    """Repeat SDA after subsampling the eligible reference baselines.

    Reference similarities are calculated once per focal subject. Within each
    repeat, the reference pool is randomly permuted without replacement, and
    each requested size uses the first ``M'`` entries of that permutation.
    Consequently, subsets are nested across sizes within a repeat.

    Results use long format with one row per subject, subset size, and repeat.
    Full-reference statistics and differences from them are included in every
    row. Benjamini-Hochberg correction is applied across subjects separately
    for each subset-size/repeat combination and once for the full analysis.
    """
    if (
            isinstance(n_repeats, (bool, np.bool_))
            or not isinstance(n_repeats, (int, np.integer))
            or n_repeats < 1
    ):
        raise ValueError("n_repeats must be a positive integer.")
    if not isinstance(timepoints_val, (int, np.integer)):
        raise TypeError("timepoints_val must be an integer.")
    if not isinstance(strict, bool) or not isinstance(naive, bool):
        raise TypeError("strict and naive must be boolean values.")

    (
        baseline,
        baseline_full,
        abx,
        post_abx_cohorts,
        keys,
        subject_ids,
    ) = _validate_dataset(dataset)
    timepoints_val = int(timepoints_val)
    if not 0 <= timepoints_val < len(post_abx_cohorts):
        raise ValueError(
            "timepoints_val must index one of the post-antibiotic cohorts."
        )

    profiles = _calculate_subject_similarity_profiles(
        baseline,
        baseline_full,
        abx,
        post_abx_cohorts,
        keys,
        subject_ids,
        timepoints_val,
        method,
        strict,
        naive,
    )
    smallest_reference_pool = min(
        profile["reference_similarities"].size
        for profile in profiles
    )
    subset_sizes = _validate_subset_sizes(
        subset_sizes,
        smallest_reference_pool,
    )

    observed_similarities = [
        profile["observed_similarity"]
        for profile in profiles
    ]
    full_reference_similarities = [
        profile["reference_similarities"]
        for profile in profiles
    ]
    full_statistics = calculate_sda_statistics(
        observed_similarities,
        full_reference_similarities,
        subject_ids=subject_ids,
        alpha=alpha,
    )
    full_statistics = full_statistics.rename(columns={
        column: f"full_{column}"
        for column in full_statistics.columns
        if column != "subject_id"
    })

    random_generator = np.random.default_rng(random_state)
    result_frames = []
    for repeat_index in range(int(n_repeats)):
        permutations = [
            random_generator.permutation(profile["reference_similarities"].size)
            for profile in profiles
        ]
        for subset_size in subset_sizes:
            selected_similarities = []
            selected_reference_ids = []
            reference_fractions = []
            for profile, permutation in zip(profiles, permutations):
                selected_indexes = permutation[:subset_size]
                selected_similarities.append(
                    profile["reference_similarities"][selected_indexes]
                )
                selected_reference_ids.append(json.dumps([
                    str(reference_id)
                    for reference_id in profile["reference_ids"][selected_indexes]
                ]))
                reference_fractions.append(
                    subset_size / profile["reference_similarities"].size
                )

            statistics = calculate_sda_statistics(
                observed_similarities,
                selected_similarities,
                subject_ids=subject_ids,
                alpha=alpha,
            )
            statistics.insert(1, "subset_size", subset_size)
            statistics.insert(2, "reference_fraction", reference_fractions)
            statistics.insert(3, "repeat", repeat_index)
            statistics.insert(4, "reference_ids", selected_reference_ids)
            result_frames.append(statistics)

    results = pd.concat(result_frames, ignore_index=True)
    results = results.merge(full_statistics, on="subject_id", how="left")
    results["normalized_rank_difference"] = (
        results["normalized_rank"] - results["full_normalized_rank"]
    )
    results["standardized_similarity_difference"] = (
        results["standardized_similarity"]
        - results["full_standardized_similarity"]
    )
    results["significance_matches_full"] = (
        results["is_significant"] == results["full_is_significant"]
    )
    results["similarity_method"] = method
    results["timepoints_val"] = timepoints_val
    results["strict"] = strict
    results["naive"] = naive
    return results
