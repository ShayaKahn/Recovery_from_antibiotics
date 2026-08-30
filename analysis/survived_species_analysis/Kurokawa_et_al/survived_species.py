from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.analysis.survived_species_analysis import (
    run_dense_time_series_survived_species_analysis,
)
from src.host_specific_recovery.io.Kurokawa_et_al_loader import load_Kurokawa_et_al_data
from src.host_specific_recovery.visualizations.plot_survived_species_analysis import plot_survived_species_analysis


SCALE = 0#1e2
SURVIVED_SPECIES_MIN_PREVALENCE = 1.0
RETURNING_TAXA_BASELINE_MIN_PREVALENCE = 0.85
MOUSE_D_BASELINE_INDEXES = range(0, 89)
MOUSE_D_ABX_INDEXES = range(89, 109)


def summarize_returning_taxa(
        samples,
        baseline_indexes,
        abx_indexes,
        baseline_min_prevalence,
        taxa_ids=None,
        last_k_abx_samples=15,
) -> pd.DataFrame:
    """Summarize baseline taxa that disappear during ABX and later return.

    Parameters
    ----------
    samples : array-like of shape (n_samples, n_taxa)
        Relative-abundance time series for one mouse, ordered by time.
    baseline_indexes, abx_indexes : range, slice, or array-like of int
        Sample-row indexes belonging to the baseline and antibiotic periods.
    baseline_min_prevalence : float
        Minimum fraction of baseline samples in which a taxon must have
        positive abundance. For example, use ``0.8`` for 80%.
    taxa_ids : array-like, optional
        Taxon identifiers in the same order as the columns in ``samples``.
        Column indexes are used when this is omitted.
    last_k_abx_samples : int, default=15
        Require taxa to be absent in the chronologically last ``k`` samples
        of the antibiotic period.

    Returns
    -------
    pandas.DataFrame
        One row per qualifying taxon that reappears after antibiotics. The
        first-detection index is an index into the original sample matrix.
        Abundance ranks are descending (rank 1 is most abundant), use average
        abundance over positive observations only, and compare against all
        taxa in the relevant time window.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array with shape samples x taxa.")
    if not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("samples must contain finite, non-negative abundances.")

    try:
        baseline_min_prevalence = float(baseline_min_prevalence)
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline_min_prevalence must be between 0 and 1.") from exc
    if not 0 < baseline_min_prevalence <= 1:
        raise ValueError("baseline_min_prevalence must be in the interval (0, 1].")

    def normalize_indexes(indexes, name):
        if isinstance(indexes, slice):
            normalized = np.arange(samples.shape[0])[indexes]
        else:
            normalized = np.asarray(list(indexes), dtype=int)
        if normalized.ndim != 1 or normalized.size == 0:
            raise ValueError(f"{name} must contain at least one sample index.")
        if np.unique(normalized).size != normalized.size:
            raise ValueError(f"{name} must not contain duplicate indexes.")
        if normalized.min() < 0 or normalized.max() >= samples.shape[0]:
            raise ValueError(f"{name} contains an index outside samples.")
        return normalized

    baseline_indexes = normalize_indexes(baseline_indexes, "baseline_indexes")
    abx_indexes = normalize_indexes(abx_indexes, "abx_indexes")
    if np.intersect1d(baseline_indexes, abx_indexes).size:
        raise ValueError("baseline_indexes and abx_indexes must not overlap.")
    if (
            isinstance(last_k_abx_samples, (bool, np.bool_))
            or not isinstance(last_k_abx_samples, (int, np.integer))
            or not 1 <= last_k_abx_samples <= abx_indexes.size
    ):
        raise ValueError(
            "last_k_abx_samples must be an integer between 1 and the number "
            "of antibiotic samples."
        )
    last_abx_indexes = np.sort(abx_indexes)[-last_k_abx_samples:]

    recovery_indexes = np.arange(abx_indexes.max() + 1, samples.shape[0])
    if recovery_indexes.size == 0:
        raise ValueError("At least one sample is required after the antibiotic period.")

    if taxa_ids is None:
        taxa_ids = np.arange(samples.shape[1])
    else:
        taxa_ids = np.asarray(list(taxa_ids), dtype=object)
        if taxa_ids.ndim != 1 or taxa_ids.size != samples.shape[1]:
            raise ValueError("taxa_ids must have one identifier per samples column.")

    baseline = samples[baseline_indexes]
    last_abx = samples[last_abx_indexes]
    baseline_prevalence = np.mean(baseline > 0, axis=0)
    candidate_mask = (
        (baseline_prevalence >= baseline_min_prevalence)
        & np.all(last_abx == 0, axis=0)
    )

    def positive_means(values):
        positive = values > 0
        return np.divide(
            np.sum(values, axis=0),
            np.sum(positive, axis=0),
            out=np.full(values.shape[1], np.nan, dtype=float),
            where=np.sum(positive, axis=0) > 0,
        )

    baseline_positive_means = positive_means(baseline)
    baseline_ranks = pd.Series(baseline_positive_means).rank(
        method="min", ascending=False, na_option="keep"
    ).to_numpy()

    rows = []
    for taxon_index in np.flatnonzero(candidate_mask):
        detected_recovery_indexes = recovery_indexes[
            samples[recovery_indexes, taxon_index] > 0
        ]
        if detected_recovery_indexes.size == 0:
            continue

        first_detection_index = int(detected_recovery_indexes[0])
        since_first_detection = samples[first_detection_index:]
        taxon_abundances = since_first_detection[:, taxon_index]
        positive_taxon_abundances = taxon_abundances[taxon_abundances > 0]

        recovery_positive_means = positive_means(since_first_detection)
        recovery_ranks = pd.Series(recovery_positive_means).rank(
            method="min", ascending=False, na_option="keep"
        ).to_numpy()

        rows.append({
            "taxon_id": taxa_ids[taxon_index],
            "taxon_index": int(taxon_index),
            "first_detection_index": first_detection_index,
            "mean_abundance_since_first_detection": float(positive_taxon_abundances.mean()),
            "mean_abundance_baseline": float(baseline[:, taxon_index].mean()),
            "baseline_abundance_rank": int(baseline_ranks[taxon_index]),
            "recovery_abundance_rank": int(recovery_ranks[taxon_index]),
            "presence_fraction_since_first_detection": float(
                np.mean(taxon_abundances > 0)
            ),
        })

    columns = [
        "taxon_id",
        "taxon_index",
        "first_detection_index",
        "mean_abundance_since_first_detection",
        "mean_abundance_baseline",
        "baseline_abundance_rank",
        "recovery_abundance_rank",
        "presence_fraction_since_first_detection",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_Kurokawa_survived_species_analysis(
        mouse_d_baseline_indexes=MOUSE_D_BASELINE_INDEXES,
        mouse_d_abx_indexes=MOUSE_D_ABX_INDEXES,
        scale=SCALE,
        survived_species_min_prevalence=SURVIVED_SPECIES_MIN_PREVALENCE,
) -> dict:
    dataset = load_Kurokawa_et_al_data(normalize=True)

    mouse_d_results = run_dense_time_series_survived_species_analysis(
        samples=dataset["mouseD_relative_abundance"],
        baseline_indexes=mouse_d_baseline_indexes,
        abx_indexes=mouse_d_abx_indexes,
        scale=scale,
        survived_species_min_prevalence=survived_species_min_prevalence,
    )

    return {
        "mouseD": mouse_d_results,
    }


def run_Kurokawa_returning_taxa_analysis(
        mouse_d_baseline_indexes=MOUSE_D_BASELINE_INDEXES,
        mouse_d_abx_indexes=MOUSE_D_ABX_INDEXES,
        baseline_min_prevalence=RETURNING_TAXA_BASELINE_MIN_PREVALENCE,
        last_k_abx_samples=15,
) -> pd.DataFrame:
    """Apply the returning-taxa analysis to Mouse D."""
    dataset = load_Kurokawa_et_al_data(normalize=True)

    return summarize_returning_taxa(
        samples=dataset["mouseD_relative_abundance"],
        baseline_indexes=mouse_d_baseline_indexes,
        abx_indexes=mouse_d_abx_indexes,
        baseline_min_prevalence=baseline_min_prevalence,
        taxa_ids=dataset["mouseD_taxa_ids"],
        last_k_abx_samples=last_k_abx_samples,
    )


def plot_Kurokawa_survived_species_trajectory(outputs: dict):
    mouse_d_outputs = outputs["mouseD"]
    if mouse_d_outputs["n_reduced_species"] == 0:
        print("No reduced survived species found for mouseD; skipping trajectory plot.")
        return None

    n_samples = mouse_d_outputs["plot_matrix"].shape[1]
    x_vals = np.arange(n_samples)
    x_labels = [" " for idx in x_vals]

    return plot_survived_species_analysis(
        {
            "results_matrix": mouse_d_outputs["plot_matrix"],
            "mean": mouse_d_outputs["plot_mean"],
        },
        x_vals=x_vals,
        x_labels=x_labels,
        xticks_fontsize=100,
        ylable_fontsize=150,
        ytick_fontsize=120,
    )


if __name__ == "__main__":
    #outputs = run_Kurokawa_survived_species_analysis()
    mouse_d_returning_taxa = run_Kurokawa_returning_taxa_analysis(last_k_abx_samples=5)
    #mouse_d_outputs = outputs["mouseD"]

    #print("mouseD results_matrix", mouse_d_outputs["results_matrix"].shape)
    #print("mouseD mean", mouse_d_outputs["mean"].shape)
    #print("mouseD n_survived_species", mouse_d_outputs["n_survived_species"])
    #print("mouseD n_reduced_species", mouse_d_outputs["n_reduced_species"])
    #print("mouseD returning taxa")
    #print(mouse_d_returning_taxa.to_string(index=False))
    #plot_Kurokawa_survived_species_trajectory(outputs)
    print(0)
