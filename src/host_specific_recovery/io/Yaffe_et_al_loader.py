import pandas as pd
import re
from typing import Dict, Optional

def load_Yaffe_et_al_data() -> Dict[str, Optional[object]]:

    file_path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/PRJNA974858/merged_metaphlan_species_table.tsv"

    metadata_path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/PRJNA974858/metadata.csv"

    metadata = pd.read_csv(metadata_path)

    df = pd.read_csv(file_path, sep="\t", index_col=0)

    run_to_sample = metadata.set_index("Run")["Sample Name"].to_dict()

    df = df.rename(columns=run_to_sample)

    pattern = re.compile(r"^(.+)_(\d+)$")

    unique_ids = sorted({
        pattern.match(str(col)).group(1)
        for col in df.columns
        if pattern.match(str(col))
    })

    def extract_id_columns(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
        pattern = re.compile(rf"^{re.escape(sample_id)}_(\d+)$")

        cols = [col for col in df.columns if pattern.match(str(col))]
        cols_sorted = sorted(cols, key=lambda col: int(pattern.match(str(col)).group(1)))

        return df[cols_sorted]

    subjects_dfs = [extract_id_columns(df, i) for i in unique_ids]

    subjects_dfs_full = [subjects_dfs[j] for j in range(len(subjects_dfs)) if subjects_dfs[j].shape[1] == 16]

    keys = sorted({
        pattern.match(str(col)).group(1)
        for col in pd.concat(subjects_dfs_full, axis=1).columns
        if pattern.match(str(col))
    })

    def collect_ith_columns(dfs: list[pd.DataFrame], i: int) -> pd.DataFrame:
        cols = [df.iloc[:, i] for df in dfs]
        return pd.concat(cols, axis=1)

    baseline_minus_63 = collect_ith_columns(subjects_dfs_full, 0) / 100.
    baseline_minus_2 = collect_ith_columns(subjects_dfs_full, 1) / 100.
    baseline_minus_1 = collect_ith_columns(subjects_dfs_full, 2) / 100.
    baseline = collect_ith_columns(subjects_dfs_full, 2) / 100.
    abx_1 = collect_ith_columns(subjects_dfs_full, 4) / 100.
    abx_2 = collect_ith_columns(subjects_dfs_full, 5) / 100.
    abx_3 = collect_ith_columns(subjects_dfs_full, 6) / 100.
    abx_4 = collect_ith_columns(subjects_dfs_full, 7) / 100.
    abx_5 = collect_ith_columns(subjects_dfs_full, 8) / 100.
    post_6 = collect_ith_columns(subjects_dfs_full, 9) / 100.
    post_7 = collect_ith_columns(subjects_dfs_full, 10) / 100.
    post_8 = collect_ith_columns(subjects_dfs_full, 11) / 100.
    post_10 = collect_ith_columns(subjects_dfs_full, 12) / 100.
    post_18 = collect_ith_columns(subjects_dfs_full, 13) / 100.
    post_28 = collect_ith_columns(subjects_dfs_full, 14) / 100.
    post_77 = collect_ith_columns(subjects_dfs_full, 15) / 100.

    # Transform to Numpy
    baseline_minus_63 = baseline_minus_63.to_numpy().T
    baseline_minus_2 = baseline_minus_2.to_numpy().T
    baseline_minus_1 = baseline_minus_1.to_numpy().T
    baseline = baseline.to_numpy().T
    abx_1 = abx_1.to_numpy().T
    abx_2 = abx_2.to_numpy().T
    abx_3 = abx_3.to_numpy().T
    abx_4 = abx_4.to_numpy().T
    abx_5 = abx_5.to_numpy().T
    post_6 = post_6.to_numpy().T
    post_7 = post_7.to_numpy().T
    post_8 = post_8.to_numpy().T
    post_10 = post_10.to_numpy().T
    post_18 = post_18.to_numpy().T
    post_28 = post_28.to_numpy().T
    post_77 = post_77.to_numpy().T

    filtered_keys = keys

    times_post_abx = [6, 7, 8, 10, 18, 28, 77]

    return {
        "abundance_table": df,

        "baseline": baseline_minus_1,
        "baseline_minus_63": baseline_minus_63,
        "baseline_minus_2": baseline_minus_2,
        "post_abx": post_77,
        "post_abx_6": post_6,
        "post_abx_7": post_7,
        "post_abx_8": post_8,
        "post_abx_10": post_10,
        "post_abx_18": post_18,
        "post_abx_28": post_28,
        "abx": abx_5,
        "abx_4": abx_4,
        "abx_3": abx_3,
        "abx_2": abx_2,
        "abx_1": abx_1,
        "baseline_full": baseline,

        "post_abx_cohorts": [
            post_6,
            post_7,
            post_8,
            post_10,
            post_18,
            post_28,
            post_77
        ],

        "times_post_abx": times_post_abx,

        "keys": keys,
        "filtered_keys": filtered_keys
    }
