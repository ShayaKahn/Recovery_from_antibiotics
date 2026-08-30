from src.host_specific_recovery.io.common import *
from typing import Dict, List, Mapping, Sequence
import os
from Bio import Phylo

def _build_filtered_keys() -> List[str]:
    return ["10", "18", "40", "57", "65", "77", "85", "108", "123", "5117", "15", "51", "72", "93", "105",
            "112", "122", "25", "38", "52", "67", "78", "89", "107", "118", "125", "140"]

def _normalize_columns(df):
    return df.div(df.sum(axis=0), axis=1)


def _align_sample_depths(
        sample_depth_raw: pd.Series,
        subject_ids: Sequence[str],
        sample_columns_by_timepoint: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Align raw sample depths to the analyzed subjects and timepoints."""
    if not sample_depth_raw.index.is_unique:
        raise ValueError("sample_depth_raw must have unique sample names.")

    subject_ids = [str(subject_id) for subject_id in subject_ids]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject_ids must be unique.")

    aligned_depths = {}
    for timepoint, sample_columns in sample_columns_by_timepoint.items():
        sample_columns = list(sample_columns)
        if len(sample_columns) != len(subject_ids):
            raise ValueError(
                f"{timepoint} must contain exactly {len(subject_ids)} samples."
            )

        sample_subject_ids = [str(name).split("_", 1)[0] for name in sample_columns]
        if sample_subject_ids != subject_ids:
            raise ValueError(
                f"The sample order for {timepoint} does not match filtered_keys."
            )

        missing_names = [
            name for name in sample_columns if name not in sample_depth_raw.index
        ]
        if missing_names:
            missing = ", ".join(map(str, missing_names))
            raise KeyError(
                f"sample_depth_raw is missing required sample(s) for "
                f"{timepoint}: {missing}."
            )

        aligned_depths[timepoint] = sample_depth_raw.loc[sample_columns].to_numpy()

    sample_depth = pd.DataFrame(aligned_depths, index=subject_ids)
    sample_depth.index.name = "subject_id"

    values = sample_depth.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All aligned sample depths must be positive and finite.")
    return sample_depth

def load_Messaoudene_et_al_data(dir="C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/Data_csv"
                                ) -> Dict[str, Optional[object]]:

    tree = Phylo.read("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_tree/"
                      "exported-tree/tree.nwk", "newick")

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
    depth = data.iloc[:, 0].sum()

    data = _normalize_columns(data)
    df_baseline_full = _normalize_columns(df_baseline_full)
    df_post_full = _normalize_columns(df_post_full)
    df_baseline = _normalize_columns(df_baseline)
    df_ABX_3 = _normalize_columns(df_ABX_3)
    df_ABX = _normalize_columns(df_ABX)
    df_post_ABX_9 = _normalize_columns(df_post_ABX_9)
    df_post_ABX_12 = _normalize_columns(df_post_ABX_12)
    df_post_ABX_16 = _normalize_columns(df_post_ABX_16)
    df_post_ABX_25 = _normalize_columns(df_post_ABX_25)
    df_post_ABX = _normalize_columns(df_post_ABX)

    # load data not rarifired
    data_no_rar_path = os.path.join(dir, 'full_ASV_table_no_rarified.csv')
    if not os.path.exists(data_no_rar_path):
        # Backward compatibility with the original misspelled filename.
        data_no_rar_path = os.path.join(dir, 'full_ASV_table_no_rerified.csv')
    data_no_rar = load_csv_df(data_no_rar_path, index_col=0)
    sample_depth_raw = data_no_rar.sum(axis=0)
    df_baseline_full_no_rar = load_csv_df(os.path.join(dir, 'baseline_full_no_rarified.csv'), index_col=0)
    df_post_full_no_rar = load_csv_df(os.path.join(dir, 'post_ABX_full_no_rarified.csv'), index_col=0)
    df_baseline_no_rar = load_csv_df(os.path.join(dir, 'baseline_subjects_no_rarified.csv'), index_col=0)
    df_ABX_3_no_rar = load_csv_df(os.path.join(dir, 'day3_subjects_no_rarified.csv'), index_col=0)
    df_ABX_no_rar = load_csv_df(os.path.join(dir, 'ABX_subjects_no_rarified.csv'), index_col=0)
    df_post_ABX_9_no_rar = load_csv_df(os.path.join(dir, 'day9_subjects_no_rarified.csv'), index_col=0)
    df_post_ABX_12_no_rar = load_csv_df(os.path.join(dir, 'day12_subjects_no_rarified.csv'), index_col=0)
    df_post_ABX_16_no_rar = load_csv_df(os.path.join(dir, 'day16_subjects_no_rarified.csv'), index_col=0)
    df_post_ABX_25_no_rar = load_csv_df(os.path.join(dir, 'day25_subjects_no_rarified.csv'), index_col=0)
    df_post_ABX_no_rar = load_csv_df(os.path.join(dir, 'post_ABX_subjects_no_rarified.csv'), index_col=0)

    data_no_rar = _normalize_columns(data_no_rar)
    df_baseline_full_no_rar = _normalize_columns(df_baseline_full_no_rar)
    df_post_full_no_rar = _normalize_columns(df_post_full_no_rar)
    df_baseline_no_rar = _normalize_columns(df_baseline_no_rar)
    df_ABX_3_no_rar = _normalize_columns(df_ABX_3_no_rar)
    df_ABX_no_rar = _normalize_columns(df_ABX_no_rar)
    df_post_ABX_9_no_rar = _normalize_columns(df_post_ABX_9_no_rar)
    df_post_ABX_12_no_rar = _normalize_columns(df_post_ABX_12_no_rar)
    df_post_ABX_16_no_rar = _normalize_columns(df_post_ABX_16_no_rar)
    df_post_ABX_25_no_rar = _normalize_columns(df_post_ABX_25_no_rar)
    df_post_ABX_no_rar = _normalize_columns(df_post_ABX_no_rar)

    filtered_keys = _build_filtered_keys()
    sample_depth = _align_sample_depths(
        sample_depth_raw,
        filtered_keys,
        {
            "baseline": df_baseline_no_rar.columns,
            "abx_3": df_ABX_3_no_rar.columns,
            "abx": df_ABX_no_rar.columns,
            "post_abx_9": df_post_ABX_9_no_rar.columns,
            "post_abx_12": df_post_ABX_12_no_rar.columns,
            "post_abx_16": df_post_ABX_16_no_rar.columns,
            "post_abx_25": df_post_ABX_25_no_rar.columns,
            "post_abx": df_post_ABX_no_rar.columns,
        },
    )

    matched_cols = []
    for key in filtered_keys:
        matched = [col for col in df_baseline_full.columns if col.startswith(f"{key}_")]
        matched_cols.extend(matched)

    remaining_cols = [col for col in df_baseline_full.columns if col not in matched_cols]

    total_cols = matched_cols + remaining_cols

    baseline_full = df_baseline_full[total_cols]
    baseline_full_no_rar = df_baseline_full_no_rar[total_cols]

    keys = [t.split('_')[0] for t in total_cols]

    baseline_full_numpy = transpose_numeric(baseline_full, norm=False)
    baseline_numpy = transpose_numeric(df_baseline, norm=False)
    ABX_3_numpy = transpose_numeric(df_ABX_3, norm=False)
    ABX_numpy = transpose_numeric(df_ABX, norm=False)
    post_ABX_9_numpy = transpose_numeric(df_post_ABX_9, norm=False)
    post_ABX_12_numpy = transpose_numeric(df_post_ABX_12, norm=False)
    post_ABX_16_numpy = transpose_numeric(df_post_ABX_16, norm=False)
    post_ABX_25_numpy = transpose_numeric(df_post_ABX_25, norm=False)
    post_ABX_numpy = transpose_numeric(df_post_ABX, norm=False)

    baseline_full_no_rar_numpy = transpose_numeric(baseline_full_no_rar, norm=False)
    baseline_no_rar_numpy = transpose_numeric(df_baseline_no_rar, norm=False)
    ABX_3_no_rar_numpy = transpose_numeric(df_ABX_3_no_rar, norm=False)
    ABX_no_rar_numpy = transpose_numeric(df_ABX_no_rar, norm=False)
    post_ABX_9_no_rar_numpy = transpose_numeric(df_post_ABX_9_no_rar, norm=False)
    post_ABX_12_no_rar_numpy = transpose_numeric(df_post_ABX_12_no_rar, norm=False)
    post_ABX_16_no_rar_numpy = transpose_numeric(df_post_ABX_16_no_rar, norm=False)
    post_ABX_25_no_rar_numpy = transpose_numeric(df_post_ABX_25_no_rar, norm=False)
    post_ABX_no_rar_numpy = transpose_numeric(df_post_ABX_no_rar, norm=False)

    times_post_abx = [4, 7, 11, 20, 32]

    # Control subjects
    def sort_lst(lst):
        return sorted(lst, key=lambda s: int(s.split('_', 1)[0]))

    def remove_prefix_numbers(items, numbers):
        prefixes = tuple(f"{int(n)}_" for n in numbers)
        return [x for x in items if not (isinstance(x, str) and x.startswith(prefixes))]

    cols_all = list(data.columns)
    cols_control = remove_prefix_numbers(sort_lst([s for s in cols_all if isinstance(s,
                                                                                     str) and s.lower().endswith(
        "_ct")]),
                                         ["83", "87"])
    cols_control_base = remove_prefix_numbers(([s for s in cols_control if "Day1_" in s]), ["83", "87"])
    cols_control_ABX3 = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day3_" in s]), ["83", "87"])
    cols_control_ABX = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day6_" in s]), ["83", "87"])
    cols_control_9 = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day9_" in s]), ["83", "87"])
    cols_control_12 = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day12_" in s]), ["83", "87"])
    cols_control_16 = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day16_" in s]), ["83", "87"])
    cols_control_25 = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day25_" in s]), ["83", "87"])
    cols_control_post = remove_prefix_numbers(sort_lst([s for s in cols_control if "Day37_" in s]), ["83", "87"])

    base_control = data[cols_control_base]
    ABX3_control = data[cols_control_ABX3]
    ABX_control = data[cols_control_ABX]
    post_9_control = data[cols_control_9]
    post_12_control = data[cols_control_12]
    post_16_control = data[cols_control_16]
    post_25_control = data[cols_control_25]
    post_control = data[cols_control_post]

    base_control_numpy = base_control.div(base_control.sum(axis=0), axis=1).to_numpy().T
    ABX3_control_numpy = ABX3_control.div(ABX3_control.sum(axis=0), axis=1).to_numpy().T
    ABX_control_numpy = ABX_control.div(ABX_control.sum(axis=0), axis=1).to_numpy().T
    post_9_control_numpy = post_9_control.div(post_9_control.sum(axis=0), axis=1).to_numpy().T
    post_12_control_numpy = post_12_control.div(post_12_control.sum(axis=0), axis=1).to_numpy().T
    post_16_control_numpy = post_16_control.div(post_16_control.sum(axis=0), axis=1).to_numpy().T
    post_25_control_numpy = post_25_control.div(post_25_control.sum(axis=0), axis=1).to_numpy().T
    post_control_numpy = post_control.div(post_control.sum(axis=0), axis=1).to_numpy().T

    return {
        "data": data,
        "baseline_full_df": baseline_full,
        "post_full_df": df_post_full,
        "baseline_df": df_baseline,
        "abx_3_df": df_ABX_3,
        "abx_df": df_ABX,
        "abx_cohorts_df": [df_ABX_3, df_ABX],
        "post_abx_9_df": df_post_ABX_9,
        "post_abx_12_df": df_post_ABX_12,
        "post_abx_16_df": df_post_ABX_16,
        "post_abx_25_df": df_post_ABX_25,
        "post_abx_df": df_post_ABX,
        "post_abx_cohorts_df": [
            df_post_ABX_9,
            df_post_ABX_12,
            df_post_ABX_16,
            df_post_ABX_25,
            df_post_ABX
        ],

        "data_no_rar": data_no_rar,
        "baseline_full_df_no_rar": baseline_full_no_rar,
        "post_full_df_no_rar": df_post_full_no_rar,
        "baseline_df_no_rar": df_baseline_no_rar,
        "abx_3_df_no_rar": df_ABX_3_no_rar,
        "abx_df_no_rar": df_ABX_no_rar,
        "abx_cohorts_df_no_rar": [df_ABX_3_no_rar, df_ABX_no_rar],
        "post_abx_9_df_no_rar": df_post_ABX_9_no_rar,
        "post_abx_12_df_no_rar": df_post_ABX_12_no_rar,
        "post_abx_16_df_no_rar": df_post_ABX_16_no_rar,
        "post_abx_25_df_no_rar": df_post_ABX_25_no_rar,
        "post_abx_df_no_rar": df_post_ABX_no_rar,
        "post_abx_cohorts_df_no_rar": [
            df_post_ABX_9_no_rar,
            df_post_ABX_12_no_rar,
            df_post_ABX_16_no_rar,
            df_post_ABX_25_no_rar,
            df_post_ABX_no_rar
        ],

        "filtered_keys": filtered_keys,
        "keys": keys,

        "baseline_full": baseline_full_numpy,
        "baseline": baseline_numpy,
        "abx_3": ABX_3_numpy,
        "abx": ABX_numpy,
        "abx_cohorts": [ABX_3_numpy, ABX_numpy],
        "post_abx_9": post_ABX_9_numpy,
        "post_abx_12": post_ABX_12_numpy,
        "post_abx_16": post_ABX_16_numpy,
        "post_abx_25": post_ABX_25_numpy,
        "post_abx": post_ABX_numpy,
        "post_abx_cohorts": [
            post_ABX_9_numpy,
            post_ABX_12_numpy,
            post_ABX_16_numpy,
            post_ABX_25_numpy,
            post_ABX_numpy
        ],

        "baseline_full_no_rar": baseline_full_no_rar_numpy,
        "baseline_no_rar": baseline_no_rar_numpy,
        "abx_3_no_rar": ABX_3_no_rar_numpy,
        "abx_no_rar": ABX_no_rar_numpy,
        "abx_cohorts_no_rar": [ABX_3_no_rar_numpy, ABX_no_rar_numpy],
        "post_abx_9_no_rar": post_ABX_9_no_rar_numpy,
        "post_abx_12_no_rar": post_ABX_12_no_rar_numpy,
        "post_abx_16_no_rar": post_ABX_16_no_rar_numpy,
        "post_abx_25_no_rar": post_ABX_25_no_rar_numpy,
        "post_abx_no_rar": post_ABX_no_rar_numpy,
        "post_abx_cohorts_no_rar": [
            post_ABX_9_no_rar_numpy,
            post_ABX_12_no_rar_numpy,
            post_ABX_16_no_rar_numpy,
            post_ABX_25_no_rar_numpy,
            post_ABX_no_rar_numpy
        ],

        "times_post_abx": times_post_abx,
        "baseline_control": base_control_numpy,
        "abx3_control": ABX3_control_numpy,
        "abx_control": ABX_control_numpy,
        "post_9_control": post_9_control_numpy,
        "post_12_control": post_12_control_numpy,
        "post_16_control": post_16_control_numpy,
        "post_25_control": post_25_control_numpy,
        "post_control": post_control_numpy,
        "post_abx_cohorts_control": [
            post_9_control_numpy,
            post_12_control_numpy,
            post_16_control_numpy,
            post_25_control_numpy,
            post_control_numpy
        ],
        
        "tree": tree,
        "depth": depth,
        "sample_depth": sample_depth,
        "sample_depth_raw": sample_depth_raw
    }

def load_Messaoudene_et_al_functional_data() -> Dict[str, Optional[object]]:

    # load data
    data = pd.read_csv('C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/picrust2_data/full_ASV_table.csv',
                       index_col=0)
    metadata = pd.read_csv("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/Metadata.csv")
    rename_map = dict(zip(metadata["Run"], metadata["Name"]))
    os.chdir("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_12_12_25_subset/picrust2_out_pipeline_strat")

    PATH_contrib_path = ("C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132_picrust2_12_12_25_subset/"
                         "picrust2_out_pipeline_strat/pathways_out/path_abun_contrib.tsv.gz")
    PATH = pd.read_csv("pathways_out/path_abun_unstrat.tsv.gz", sep="\t", index_col=0)

    return {
        'metadata': metadata,
        'data': data,
        'rename_map': rename_map,
        'PATH_contrib_path': PATH_contrib_path,
        'PATH': PATH
    }
