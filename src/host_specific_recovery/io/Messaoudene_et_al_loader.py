from src.host_specific_recovery.io.common import *
from typing import Dict, List
import os

def _build_filtered_keys() -> List[str]:
    return ["10", "18", "40", "57", "65", "77", "85", "108", "123", "5117", "15", "51", "72", "93", "105",
            "112", "122", "25", "38", "52", "67", "78", "89", "107", "118", "125", "140"]

def load_Messaoudene_et_al_data(
        dir="C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/Data_csv") -> Dict[str, Optional[object]]:

    # load data
    data = load_csv_df(os.path.join(dir, 'full_ASV_table.csv'), index_col=0)
    df_baseline_full = load_csv_df(os.path.join(dir, 'baseline_full.csv'), index_col=0)
    df_post_full = load_csv_df(os.path.join(dir, 'post_ABX_full.csv'), index_col=0)
    df_baseline = load_csv_df(os.path.join(dir, 'baseline_subjects.csv'), index_col=0)
    df_ABX_3 = load_csv_df(os.path.join(dir, 'day3_subjects.csv'), index_col=0)
    df_ABX = load_csv_df(os.path.join(dir, 'ABX_subjects.csv'), index_col=0)
    df_post_ABX_9 = load_csv_df(os.path.join(dir, 'day9_subjects.csv'), index_col=0)
    df_post_ABX_12 = load_csv_df(os.path.join(dir, 'day12_subjects.csv'), index_col=0)
    df_post_ABX_16 = load_csv_df(os.path.join(dir, 'day16_subjects.csv'), index_col=0)
    df_post_ABX_25 = load_csv_df(os.path.join(dir, 'day25_subjects.csv'), index_col=0)
    df_post_ABX = load_csv_df(os.path.join(dir, 'post_ABX_subjects.csv'), index_col=0)

    filtered_keys = _build_filtered_keys()

    matched_cols = []
    for key in filtered_keys:
        matched = [col for col in df_baseline_full.columns if col.startswith(f"{key}_")]
        matched_cols.extend(matched)

    remaining_cols = [col for col in df_baseline_full.columns if col not in matched_cols]

    total_cols = matched_cols + remaining_cols

    baseline_full = df_baseline_full[total_cols]

    keys = [t.split('_')[0] for t in total_cols]

    # normalization
    baseline_full_numpy = transpose_numeric(baseline_full, norm=True)
    baseline_numpy = transpose_numeric(df_baseline, norm=True)
    ABX_3_numpy = transpose_numeric(df_ABX_3, norm=True)
    ABX_numpy = transpose_numeric(df_ABX, norm=True)
    post_ABX_9_numpy = transpose_numeric(df_post_ABX_9, norm=True)
    post_ABX_12_numpy = transpose_numeric(df_post_ABX_12, norm=True)
    post_ABX_16_numpy = transpose_numeric(df_post_ABX_16, norm=True)
    post_ABX_25_numpy = transpose_numeric(df_post_ABX_25, norm=True)
    post_ABX_numpy = transpose_numeric(df_post_ABX, norm=True)

    return {
        "data": data,
        "baseline_full": baseline_full,
        "post_full": df_post_full,
        "baseline": df_baseline,
        "abx_3": df_ABX_3,
        "abx": df_ABX,
        "post_abx_9": df_post_ABX_9,
        "post_abx_12": df_post_ABX_12,
        "post_abx_16": df_post_ABX_16,
        "post_abx_25": df_post_ABX_25,
        "post_abx": df_post_ABX,
        "filtered_keys": filtered_keys,
        "keys": keys,
        "baseline_full_numpy": baseline_full_numpy,
        "baseline_numpy": baseline_numpy,
        "abx_3_numpy": ABX_3_numpy,
        "abx_numpy": ABX_numpy,
        "abx_cohorts": [ABX_3_numpy, ABX_numpy],
        "post_abx_9_numpy": post_ABX_9_numpy,
        "post_abx_12_numpy": post_ABX_12_numpy,
        "post_abx_16_numpy": post_ABX_16_numpy,
        "post_abx_25_numpy": post_ABX_25_numpy,
        "post_abx_numpy": post_ABX_numpy,
        "post_abx_cohorts": [
            post_ABX_9_numpy,
            post_ABX_12_numpy,
            post_ABX_16_numpy,
            post_ABX_25_numpy,
            post_ABX_numpy
        ]
    }

def load_Messaoudene_et_al_functional_data() -> Dict[str, Optional[object]]:

    # load data
    metadata = pd.read_csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/Metadata.csv")
    rename_map = dict(zip(metadata["Run"], metadata["Name"]))
    os.chdir("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_12_12_25_subset/picrust2_out_pipeline_strat")

    PATH_contrib_path = ("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_12_12_25_subset/"
                         "picrust2_out_pipeline_strat/pathways_out/path_abun_contrib.tsv.gz")
    PATH = pd.read_csv("pathways_out/path_abun_unstrat.tsv.gz", sep="\t", index_col=0)

    return {
        'rename_map': rename_map,
        'PATH_contrib_path': PATH_contrib_path,
        'PATH': PATH
    }
