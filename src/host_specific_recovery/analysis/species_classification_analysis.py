import pandas as pd
import numpy as np
from src.host_specific_recovery.utils.general_utils import subset


def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def run_species_classification_analysis(dataset: dict):
    baseline = dataset["baseline"]
    abx = dataset["abx"]
    post_abx_cohorts = dataset["post_abx_cohorts"]
    names = dataset["filtered_keys"]

    strict = True
    num_post_abx = len(post_abx_cohorts)
    timepoints = np.arange(1, num_post_abx + 1)

    classes = {"Survived": 1, **{f"Returned {t}": t + 1 for t in timepoints}}
    survived_mat = []
    returned = []

    for i, (base_sample, abx_sample) in enumerate(zip(baseline, abx)):
        survived_mat.append(
            (base_sample != 0)
            & (post_abx_cohorts[-1][i, :] != 0)
            & (abx_sample != 0)
        )

        post_matrix = np.vstack(
            [post_abx_cohorts[tp][i, :] for tp in range(num_post_abx)]
        )
        returned_species = subset(
            post_matrix,
            base_sample,
            abx_sample,
            strict,
            new=False,
        )
        returned.append(returned_species)

    class_mat = np.zeros(baseline.shape)
    class_mat[np.vstack(survived_mat)] = classes["Survived"]
    for t in timepoints:
        class_mat[np.vstack([species[t - 1] for species in returned])] = classes[f"Returned {t}"]

    df = pd.DataFrame(class_mat.T, columns=names).astype(int)

    classifications = df[names]
    survived_counts = classifications.eq(classes["Survived"]).sum(axis=1)
    classified_counts = classifications.ne(0).sum(axis=1)
    returned_mask = classifications.ne(0) & classifications.ne(classes["Survived"])
    returned_counts = returned_mask.sum(axis=1)

    df["ps"] = _safe_divide(
        survived_counts.to_numpy(dtype=float),
        classified_counts.to_numpy(dtype=float),
    )
    df["counts"] = classified_counts
    df["counts returned"] = returned_counts

    p_return_cols = []
    returned_count_values = returned_counts.to_numpy(dtype=float)
    for t in timepoints:
        column = f"p_{t}"
        p_return_cols.append(column)

        return_counts_at_t = classifications.eq(classes[f"Returned {t}"]).sum(axis=1)
        df[column] = _safe_divide(
            return_counts_at_t.to_numpy(dtype=float),
            returned_count_values,
        )

    times = np.array(dataset["times_post_abx"])
    df["t_mean"] = (df.loc[:, p_return_cols] * times).mean(axis=1)

    return df
