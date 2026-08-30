from collections.abc import Mapping, Sequence

import numpy as np


def run_new_species_first_detection_analysis(
        dataset: Mapping,
        subject_ids: Sequence | None = None,
) -> dict[str, np.ndarray]:
    """Count species at their first post-ABX detection for every subject.

    A species is considered new when it is undetected at baseline and ABX.
    It is assigned to the first post-ABX timepoint at which its abundance is
    positive, regardless of whether it remains detected at later timepoints.
    Species that are never detected post-ABX are not counted.

    Parameters
    ----------
    dataset : mapping
        Dataset containing ``baseline``, ``abx``, ``post_abx_cohorts``, and
        ``times_post_abx``.
    subject_ids : sequence, optional
        Subject labels in matrix-row order. When omitted, labels are inferred
        from ``filtered_keys``, ``subject_ids``, or ``keys`` in the dataset;
        otherwise integer row indices are used.

    Returns
    -------
    dict
        ``new_species_counts`` is a subjects-by-timepoints matrix,
        ``total_new_species_counts`` contains one total per subject,
        ``total_new_species_counts_excluding_last`` contains the corresponding
        total without species first detected at the final timepoint,
        ``colonized_new_species_counts`` contains the number per subject that
        remained detected at every timepoint after first appearing (excluding
        species first detected at the final timepoint),
        ``subject_ids`` identifies the rows, and ``times_post_abx`` identifies
        the columns.
    """
    if not isinstance(dataset, Mapping):
        raise TypeError("dataset must be a mapping.")

    required_keys = {
        "baseline",
        "abx",
        "post_abx_cohorts",
        "times_post_abx",
    }
    missing_keys = required_keys.difference(dataset)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"dataset is missing required field(s): {missing}.")

    baseline = _as_abundance_matrix(dataset["baseline"], "baseline")
    abx = _as_abundance_matrix(dataset["abx"], "abx")
    if abx.shape != baseline.shape:
        raise ValueError("abx must have the same shape as baseline.")

    post_abx_cohorts = list(dataset["post_abx_cohorts"])
    if not post_abx_cohorts:
        raise ValueError("post_abx_cohorts must contain at least one timepoint.")
    post_abx_matrices = []
    for index, cohort in enumerate(post_abx_cohorts):
        matrix = _as_abundance_matrix(cohort, f"post_abx_cohorts[{index}]")
        if matrix.shape != baseline.shape:
            raise ValueError(
                f"post_abx_cohorts[{index}] must have the same shape as baseline."
            )
        post_abx_matrices.append(matrix)

    times_post_abx = np.asarray(dataset["times_post_abx"], dtype=float)
    if (
            times_post_abx.ndim != 1
            or times_post_abx.size != len(post_abx_matrices)
    ):
        raise ValueError(
            "times_post_abx must contain one value per post-ABX timepoint."
        )
    if (
            not np.all(np.isfinite(times_post_abx))
            or np.any(times_post_abx < 0)
            or np.any(np.diff(times_post_abx) <= 0)
    ):
        raise ValueError(
            "times_post_abx must be finite, non-negative, and strictly increasing."
        )

    n_subjects = baseline.shape[0]
    resolved_subject_ids = _resolve_subject_ids(dataset, subject_ids, n_subjects)

    post_abx = np.stack(post_abx_matrices, axis=1)
    detected_post_abx = post_abx > 0
    detected_at_earlier_timepoint = np.concatenate(
        [
            np.zeros_like(detected_post_abx[:, :1, :], dtype=bool),
            np.logical_or.accumulate(detected_post_abx[:, :-1, :], axis=1),
        ],
        axis=1,
    )
    absent_before_post_abx = (baseline == 0) & (abx == 0)
    first_detection_masks = (
        absent_before_post_abx[:, np.newaxis, :]
        & detected_post_abx
        & ~detected_at_earlier_timepoint
    )

    new_species_counts = first_detection_masks.sum(axis=2, dtype=int)
    total_new_species_counts = new_species_counts.sum(axis=1, dtype=int)
    total_new_species_counts_excluding_last = new_species_counts[:, :-1].sum(
        axis=1,
        dtype=int,
    )
    detected_from_timepoint_onward = np.logical_and.accumulate(
        detected_post_abx[:, ::-1, :],
        axis=1,
    )[:, ::-1, :]
    colonized_first_detection_masks = (
        first_detection_masks[:, :-1, :]
        & detected_from_timepoint_onward[:, :-1, :]
    )
    colonized_new_species_counts = colonized_first_detection_masks.sum(
        axis=(1, 2),
        dtype=int,
    )

    return {
        "subject_ids": resolved_subject_ids,
        "times_post_abx": times_post_abx,
        "new_species_counts": new_species_counts,
        "total_new_species_counts": total_new_species_counts,
        "total_new_species_counts_excluding_last": (
            total_new_species_counts_excluding_last
        ),
        "colonized_new_species_counts": colonized_new_species_counts,
    }


def _as_abundance_matrix(values, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D subjects-by-species matrix.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError(f"{name} must contain finite, non-negative abundances.")
    return matrix


def _resolve_subject_ids(
        dataset: Mapping,
        subject_ids: Sequence | None,
        n_subjects: int,
) -> np.ndarray:
    if subject_ids is None:
        for key in ("filtered_keys", "subject_ids", "keys"):
            if key in dataset and len(dataset[key]) == n_subjects:
                subject_ids = dataset[key]
                break
        else:
            subject_ids = np.arange(n_subjects)

    labels = np.asarray(list(subject_ids), dtype=object)
    if labels.ndim != 1 or labels.size != n_subjects:
        raise ValueError(f"subject_ids must contain exactly {n_subjects} values.")
    if np.unique(labels).size != labels.size:
        raise ValueError("subject_ids must contain unique values.")
    return labels
