from typing import Dict, List
import pandas as pd

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


def load_Sewunet_et_al_data() -> Dict[str, Optional[object]]:

    # load data
    data = load_csv_df(r"C:/Users/USER/OneDrive/Desktop/Antibiotics/Effect of tebipenem/Appendix_2_p5.csv")
    data_columns = data.columns.tolist()
    data_columns.remove('#row.nr')
    data_columns.remove('taxon')
    data = data[data_columns]
    data.set_index('otu.id', inplace=True)

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

        "baseline": baseline_filtered,
        "post_abx": post_ABX_filtered,
        "post_abx_14": post_ABX_14_filtered,
        "post_abx_21": post_ABX_21_filtered,
        "post_abx_90": post_ABX_90_filtered,
        "abx": ABX_filtered,
        "abx_7": ABX_7_filtered,
        "baseline_full": baseline_full,

        "baseline_df": df_baseline_filtered,
        "post_abx_df": df_post_ABX_filtered,
        "post_abx_14_df": df_post_ABX_14_filtered,
        "post_abx_21_df": df_post_ABX_21_filtered,
        "post_abx_90_df": df_post_ABX_90_filtered,
        "abx_df": df_ABX_filtered,
        "abx_7_df": df_ABX_7_filtered,

        "df_baseline": df_baseline_filtered,
        "df_post_ABX": df_post_ABX_filtered,
        "df_post_ABX_14": df_post_ABX_14_filtered,
        "df_post_ABX_21": df_post_ABX_21_filtered,
        "df_post_ABX_90": df_post_ABX_90_filtered,
        "df_ABX": df_ABX_filtered,
        "df_ABX_7": df_ABX_7_filtered,

        "post_abx_cohorts": [
            post_ABX_14_filtered,
            post_ABX_21_filtered,
            post_ABX_90_filtered,
            post_ABX_filtered],

        "post_abx_cohorts_df": [
            df_post_ABX_14_filtered,
            df_post_ABX_21_filtered,
            df_post_ABX_90_filtered,
            df_post_ABX_filtered],

        "times_post_abx": times_post_abx,

        "keys": keys,
        "filtered_keys": filtered_keys
    }
