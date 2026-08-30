from typing import Dict, List, Sequence
import pandas as pd
import numpy as np

from src.host_specific_recovery.io.common import *

def _get_keep_indices(indices_to_remove: list[int], n_samples: int) -> list[int]:
    remove_set = set(indices_to_remove)

    if any(i < 0 or i >= n_samples for i in remove_set):
        raise ValueError("indices_to_remove contains out-of-bounds values")

    return [i for i in range(n_samples) if i not in remove_set]


def _build_keys() -> List[str]:
    return ['2 AMC', '3 TBPM', '4 AMC', '6 TBPM', '7 TBPM', '8 TBPM', '11 TBPM', '13 AMC', '14 TBPM', '15 AMC',
            '17 AMC', '18 AMC', '19 TBPM', '20 AMC', '21 TBPM', '23 AMC', '24 AMC', '29 TBPM', '31 TBPM', '33 AMC',
            '35 AMC', '36 TBPM', '37 AMC', '40 TBPM', '42 TBPM', '45 TBPM', '47 AMC', '49 AMC', '50 TBPM']


def _build_filtered_keys(keys: List[str], keys_to_remove: list[str]) -> List[str]:
    return [k for k in keys if k not in keys_to_remove]


def _filtered_matrix_to_df(matrix, source_df, sample_ids):
    return pd.DataFrame(matrix.T, index=source_df.index, columns=sample_ids)


def _align_sample_depths(sample_depth_raw: pd.Series, subject_ids: Sequence[str]) -> pd.DataFrame:
    """Align raw sample depths to the loader's subject and timepoint names."""
    if not sample_depth_raw.index.is_unique:
        raise ValueError("sample_depth_raw must have unique sample names.")

    timepoint_suffixes = {
        "baseline": "D.1",
        "abx_7": "D7",
        "abx": "D10",
        "post_abx_14": "D14",
        "post_abx_21": "D21",
        "post_abx_90": "D90",
        "post_abx": "D180",
    }
    raw_names_by_subject = {}
    for subject_id in subject_ids:
        subject_number = int(str(subject_id).split(maxsplit=1)[0])
        raw_prefix = f"S{subject_number:04d}"
        raw_names_by_subject[subject_id] = {
            timepoint: f"{raw_prefix}{suffix}"
            for timepoint, suffix in timepoint_suffixes.items()
        }

    required_names = [
        raw_name
        for subject_names in raw_names_by_subject.values()
        for raw_name in subject_names.values()
    ]
    missing_names = [name for name in required_names if name not in sample_depth_raw.index]
    if missing_names:
        missing = ", ".join(missing_names)
        raise KeyError(f"sample_depth_raw is missing required sample(s): {missing}.")

    sample_depth = pd.DataFrame(
        {
            subject_id: {
                timepoint: sample_depth_raw.at[raw_name]
                for timepoint, raw_name in subject_names.items()
            }
            for subject_id, subject_names in raw_names_by_subject.items()
        }
    ).T
    sample_depth.index.name = "subject_id"
    sample_depth = sample_depth.loc[:, list(timepoint_suffixes)]

    values = sample_depth.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All aligned sample depths must be positive and finite.")
    return sample_depth


def load_Sewunet_et_al_data() -> Dict[str, Optional[object]]:

    # load data
    data = load_csv_df(r"C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/Appendix_2_p5.csv")
    data_columns = data.columns.tolist()
    data_columns.remove('#row.nr')
    data_columns.remove('taxon')
    data = data[data_columns]
    data.set_index('otu.id', inplace=True)
    sample_depth_raw = data.sum(axis=0)

    df_baseline_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/baseline_no_rarified.csv')
    df_post_ABX_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_no_rarified.csv')
    df_post_ABX_90_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_90_no_rarified.csv')
    df_post_ABX_21_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_21_no_rarified.csv')
    df_post_ABX_14_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_14_no_rarified.csv')
    df_ABX_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_no_rarified.csv')
    df_ABX_7_no_rar = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_7_no_rarified.csv')

    # Transform to normalized numpy arrays.
    baseline_numpy_no_rar = transpose_numeric(df_baseline_no_rar, norm=True)
    post_ABX_numpy_no_rar = transpose_numeric(df_post_ABX_no_rar, norm=True)
    post_ABX_90_numpy_no_rar = transpose_numeric(df_post_ABX_90_no_rar, norm=True)
    post_ABX_21_numpy_no_rar = transpose_numeric(df_post_ABX_21_no_rar, norm=True)
    post_ABX_14_numpy_no_rar = transpose_numeric(df_post_ABX_14_no_rar, norm=True)
    ABX_numpy_no_rar = transpose_numeric(df_ABX_no_rar, norm=True)
    ABX_7_numpy_no_rar = transpose_numeric(df_ABX_7_no_rar, norm=True)

    # Remove outliers
    indices_to_remove = [10]
    keep_indices = _get_keep_indices(indices_to_remove, baseline_numpy_no_rar.shape[0])

    baseline_filtered_no_rar = baseline_numpy_no_rar[keep_indices, :]
    post_ABX_filtered_no_rar = post_ABX_numpy_no_rar[keep_indices, :]
    post_ABX_90_filtered_no_rar = post_ABX_90_numpy_no_rar[keep_indices, :]
    post_ABX_21_filtered_no_rar = post_ABX_21_numpy_no_rar[keep_indices, :]
    post_ABX_14_filtered_no_rar = post_ABX_14_numpy_no_rar[keep_indices, :]
    ABX_filtered_no_rar = ABX_numpy_no_rar[keep_indices, :]
    ABX_7_filtered_no_rar = ABX_7_numpy_no_rar[keep_indices, :]

    baseline_full_no_rar = baseline_numpy_no_rar.copy()

    keys = _build_keys()
    keys_to_remove = ['17 AMC']
    filtered_keys = _build_filtered_keys(keys, keys_to_remove)

    keys = list_to_numpy(keys)
    filtered_keys = list_to_numpy(filtered_keys)
    sample_depth = _align_sample_depths(sample_depth_raw, filtered_keys)

    df_baseline_filtered_no_rar = _filtered_matrix_to_df(baseline_filtered_no_rar, df_baseline_no_rar, filtered_keys)
    df_post_ABX_filtered_no_rar = _filtered_matrix_to_df(post_ABX_filtered_no_rar, df_post_ABX_no_rar, filtered_keys)
    df_post_ABX_14_filtered_no_rar = _filtered_matrix_to_df(post_ABX_14_filtered_no_rar, df_post_ABX_14_no_rar, filtered_keys)
    df_post_ABX_21_filtered_no_rar = _filtered_matrix_to_df(post_ABX_21_filtered_no_rar, df_post_ABX_21_no_rar, filtered_keys)
    df_post_ABX_90_filtered_no_rar = _filtered_matrix_to_df(post_ABX_90_filtered_no_rar, df_post_ABX_90_no_rar, filtered_keys)
    df_ABX_filtered_no_rar = _filtered_matrix_to_df(ABX_filtered_no_rar, df_ABX_no_rar, filtered_keys)
    df_ABX_7_filtered_no_rar = _filtered_matrix_to_df(ABX_7_filtered_no_rar, df_ABX_7_no_rar, filtered_keys)

    # load data rarified
    data_rar = load_csv_df(r"C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/data_rarified.csv")
    data_columns_rar = data_rar.columns.tolist()
    data_columns_rar.remove('#row.nr')
    data_columns_rar.remove('taxon')
    data_rar = data_rar[data_columns_rar]
    data_rar.set_index('otu.id', inplace=True)
    depth = data_rar.iloc[:, 0].sum()

    df_baseline = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/baseline.csv')
    df_post_ABX = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX.csv')
    df_post_ABX_90 = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_90.csv')
    df_post_ABX_21 = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_21.csv')
    df_post_ABX_14 = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/post_ABX_14.csv')
    df_ABX = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX.csv')
    df_ABX_7 = load_csv_df('C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/ABX_7.csv')

    # Transform to normalized numpy arrays.
    baseline_numpy = transpose_numeric(df_baseline, norm=True)
    post_ABX_numpy = transpose_numeric(df_post_ABX, norm=True)
    post_ABX_90_numpy = transpose_numeric(df_post_ABX_90, norm=True)
    post_ABX_21_numpy = transpose_numeric(df_post_ABX_21, norm=True)
    post_ABX_14_numpy = transpose_numeric(df_post_ABX_14, norm=True)
    ABX_numpy = transpose_numeric(df_ABX, norm=True)
    ABX_7_numpy = transpose_numeric(df_ABX_7, norm=True)

    # Remove outliers
    indices_to_remove = [10]
    keep_indices = _get_keep_indices(indices_to_remove, baseline_numpy.shape[0])

    baseline_filtered = baseline_numpy[keep_indices, :]
    post_ABX_filtered = post_ABX_numpy[keep_indices, :]
    post_ABX_90_filtered = post_ABX_90_numpy[keep_indices, :]
    post_ABX_21_filtered = post_ABX_21_numpy[keep_indices, :]
    post_ABX_14_filtered = post_ABX_14_numpy[keep_indices, :]
    ABX_filtered = ABX_numpy[keep_indices, :]
    ABX_7_filtered = ABX_7_numpy[keep_indices, :]

    baseline_full = baseline_numpy.copy()

    keys = _build_keys()
    keys_to_remove = ['17 AMC']
    filtered_keys = _build_filtered_keys(keys, keys_to_remove)

    keys = list_to_numpy(keys)
    filtered_keys = list_to_numpy(filtered_keys)

    df_baseline_filtered = _filtered_matrix_to_df(baseline_filtered, df_baseline, filtered_keys)
    df_post_ABX_filtered = _filtered_matrix_to_df(post_ABX_filtered, df_post_ABX, filtered_keys)
    df_post_ABX_14_filtered = _filtered_matrix_to_df(post_ABX_14_filtered, df_post_ABX_14, filtered_keys)
    df_post_ABX_21_filtered = _filtered_matrix_to_df(post_ABX_21_filtered, df_post_ABX_21, filtered_keys)
    df_post_ABX_90_filtered = _filtered_matrix_to_df(post_ABX_90_filtered, df_post_ABX_90, filtered_keys)
    df_ABX_filtered = _filtered_matrix_to_df(ABX_filtered, df_ABX, filtered_keys)
    df_ABX_7_filtered = _filtered_matrix_to_df(ABX_7_filtered, df_ABX_7, filtered_keys)

    times_post_abx = [4, 11, 80, 170]

    return {
        "abundance_table": data,
        "abundance_table_rarified": data_rar,

        "baseline": baseline_filtered,
        "post_abx": post_ABX_filtered,
        "post_abx_14": post_ABX_14_filtered,
        "post_abx_21": post_ABX_21_filtered,
        "post_abx_90": post_ABX_90_filtered,
        "abx": ABX_filtered,
        "abx_7": ABX_7_filtered,
        "baseline_full": baseline_full,

        "baseline_no_rar": baseline_filtered_no_rar,
        "post_abx_no_rar": post_ABX_filtered_no_rar,
        "post_abx_14_no_rar": post_ABX_14_filtered_no_rar,
        "post_abx_21_no_rar": post_ABX_21_filtered_no_rar,
        "post_abx_90_no_rar": post_ABX_90_filtered_no_rar,
        "abx_no_rar": ABX_filtered_no_rar,
        "abx_7_no_rar": ABX_7_filtered_no_rar,
        "baseline_full_no_rar": baseline_full_no_rar,

        "baseline_df": df_baseline_filtered,
        "post_abx_df": df_post_ABX_filtered,
        "post_abx_14_df": df_post_ABX_14_filtered,
        "post_abx_21_df": df_post_ABX_21_filtered,
        "post_abx_90_df": df_post_ABX_90_filtered,
        "abx_df": df_ABX_filtered,
        "abx_7_df": df_ABX_7_filtered,

        "baseline_df_no_rar": df_baseline_filtered_no_rar,
        "post_abx_df_no_rar": df_post_ABX_filtered_no_rar,
        "post_abx_14_df_no_rar": df_post_ABX_14_filtered_no_rar,
        "post_abx_21_df_no_rar": df_post_ABX_21_filtered_no_rar,
        "post_abx_90_df_no_rar": df_post_ABX_90_filtered_no_rar,
        "abx_df_no_rar": df_ABX_filtered_no_rar,
        "abx_7_df_no_rar": df_ABX_7_filtered_no_rar,

        "df_baseline": df_baseline_filtered,
        "df_post_ABX": df_post_ABX_filtered,
        "df_post_ABX_14": df_post_ABX_14_filtered,
        "df_post_ABX_21": df_post_ABX_21_filtered,
        "df_post_ABX_90": df_post_ABX_90_filtered,
        "df_ABX": df_ABX_filtered,
        "df_ABX_7": df_ABX_7_filtered,

        "df_baseline_no_rar": df_baseline_filtered_no_rar,
        "df_post_ABX_no_rar": df_post_ABX_filtered_no_rar,
        "df_post_ABX_14_no_rar": df_post_ABX_14_filtered_no_rar,
        "df_post_ABX_21_no_rar": df_post_ABX_21_filtered_no_rar,
        "df_post_ABX_90_no_rar": df_post_ABX_90_filtered_no_rar,
        "df_ABX_no_rar": df_ABX_filtered_no_rar,
        "df_ABX_7_no_rar": df_ABX_7_filtered_no_rar,

        "post_abx_cohorts": [
            post_ABX_14_filtered,
            post_ABX_21_filtered,
            post_ABX_90_filtered,
            post_ABX_filtered],

        "post_abx_cohorts_no_rar": [
            post_ABX_14_filtered_no_rar,
            post_ABX_21_filtered_no_rar,
            post_ABX_90_filtered_no_rar,
            post_ABX_filtered_no_rar],

        "post_abx_cohorts_df": [
            df_post_ABX_14_filtered,
            df_post_ABX_21_filtered,
            df_post_ABX_90_filtered,
            df_post_ABX_filtered],

        "post_abx_cohorts_df_no_rar": [
            df_post_ABX_14_filtered_no_rar,
            df_post_ABX_21_filtered_no_rar,
            df_post_ABX_90_filtered_no_rar,
            df_post_ABX_filtered_no_rar],

        "times_post_abx": times_post_abx,

        "keys": keys,
        "filtered_keys": filtered_keys,
        "depth": depth,
        "sample_depth": sample_depth,
        "sample_depth_raw": sample_depth_raw
    }
