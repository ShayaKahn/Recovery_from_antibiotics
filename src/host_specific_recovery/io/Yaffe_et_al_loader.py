import pandas as pd
import re
from typing import Dict, Optional

def load_Yaffe_et_al_data() -> Dict[str, Optional[object]]:

    file_path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/PRJNA974858/merged_metaphlan_species_table.tsv"

    metadata_path = "C:/Users/USER/OneDrive/Desktop/Antibiotics/PRJNA974858/metadata.csv"
    control_prefixes = ("CAA", "CAK", "CAM", "CAC", "CAN")

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

    subjects_dfs = [extract_id_columns(df, sample_id) for sample_id in unique_ids]
    subjects_full = [
        (sample_id, subject_df)
        for sample_id, subject_df in zip(unique_ids, subjects_dfs)
        if subject_df.shape[1] == 16
    ]

    control_subjects = [
        (sample_id, subject_df)
        for sample_id, subject_df in subjects_full
        if sample_id.startswith(control_prefixes)
    ]
    treatment_subjects = [
        (sample_id, subject_df)
        for sample_id, subject_df in subjects_full
        if not sample_id.startswith(control_prefixes)
    ]
    ordered_subjects = treatment_subjects + control_subjects

    keys = [sample_id for sample_id, _ in ordered_subjects]
    keys_control = [sample_id for sample_id, _ in control_subjects]

    def collect_ith_columns(dfs: list[pd.DataFrame], i: int) -> pd.DataFrame:
        cols = [df.iloc[:, i] for df in dfs]
        if not cols:
            return pd.DataFrame(index=df.index)
        return pd.concat(cols, axis=1)

    def build_timepoint_data(subjects: list[tuple[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
        subject_dfs = [subject_df for _, subject_df in subjects]
        return {
            "baseline_minus_63_df": collect_ith_columns(subject_dfs, 0) / 100.,
            "baseline_minus_2_df": collect_ith_columns(subject_dfs, 1) / 100.,
            "baseline_minus_1_df": collect_ith_columns(subject_dfs, 2) / 100.,
            "baseline_df": collect_ith_columns(subject_dfs, 2) / 100.,
            "abx_1_df": collect_ith_columns(subject_dfs, 4) / 100.,
            "abx_2_df": collect_ith_columns(subject_dfs, 5) / 100.,
            "abx_3_df": collect_ith_columns(subject_dfs, 6) / 100.,
            "abx_4_df": collect_ith_columns(subject_dfs, 7) / 100.,
            "abx_5_df": collect_ith_columns(subject_dfs, 8) / 100.,
            "post_6_df": collect_ith_columns(subject_dfs, 9) / 100.,
            "post_7_df": collect_ith_columns(subject_dfs, 10) / 100.,
            "post_8_df": collect_ith_columns(subject_dfs, 11) / 100.,
            "post_10_df": collect_ith_columns(subject_dfs, 12) / 100.,
            "post_18_df": collect_ith_columns(subject_dfs, 13) / 100.,
            "post_28_df": collect_ith_columns(subject_dfs, 14) / 100.,
            "post_77_df": collect_ith_columns(subject_dfs, 15) / 100.,
        }

    treatment_data = build_timepoint_data(treatment_subjects)
    control_data = build_timepoint_data(control_subjects)
    full_data = build_timepoint_data(ordered_subjects)

    baseline_minus_63_df = treatment_data["baseline_minus_63_df"]
    baseline_minus_2_df = treatment_data["baseline_minus_2_df"]
    baseline_minus_1_df = treatment_data["baseline_minus_1_df"]
    baseline_df = treatment_data["baseline_df"]
    baseline_full_df = full_data["baseline_df"]
    abx_1_df = treatment_data["abx_1_df"]
    abx_2_df = treatment_data["abx_2_df"]
    abx_3_df = treatment_data["abx_3_df"]
    abx_4_df = treatment_data["abx_4_df"]
    abx_5_df = treatment_data["abx_5_df"]
    post_6_df = treatment_data["post_6_df"]
    post_7_df = treatment_data["post_7_df"]
    post_8_df = treatment_data["post_8_df"]
    post_10_df = treatment_data["post_10_df"]
    post_18_df = treatment_data["post_18_df"]
    post_28_df = treatment_data["post_28_df"]
    post_77_df = treatment_data["post_77_df"]

    baseline_minus_63_df_control = control_data["baseline_minus_63_df"]
    baseline_minus_2_df_control = control_data["baseline_minus_2_df"]
    baseline_minus_1_df_control = control_data["baseline_minus_1_df"]
    baseline_df_control = control_data["baseline_df"]
    abx_1_df_control = control_data["abx_1_df"]
    abx_2_df_control = control_data["abx_2_df"]
    abx_3_df_control = control_data["abx_3_df"]
    abx_4_df_control = control_data["abx_4_df"]
    abx_5_df_control = control_data["abx_5_df"]
    post_6_df_control = control_data["post_6_df"]
    post_7_df_control = control_data["post_7_df"]
    post_8_df_control = control_data["post_8_df"]
    post_10_df_control = control_data["post_10_df"]
    post_18_df_control = control_data["post_18_df"]
    post_28_df_control = control_data["post_28_df"]
    post_77_df_control = control_data["post_77_df"]

    # Transform to Numpy
    baseline_minus_63 = baseline_minus_63_df.to_numpy().T
    baseline_minus_2 = baseline_minus_2_df.to_numpy().T
    baseline_minus_1 = baseline_minus_1_df.to_numpy().T
    baseline = baseline_df.to_numpy().T
    abx_1 = abx_1_df.to_numpy().T
    abx_2 = abx_2_df.to_numpy().T
    abx_3 = abx_3_df.to_numpy().T
    abx_4 = abx_4_df.to_numpy().T
    abx_5 = abx_5_df.to_numpy().T
    post_6 = post_6_df.to_numpy().T
    post_7 = post_7_df.to_numpy().T
    post_8 = post_8_df.to_numpy().T
    post_10 = post_10_df.to_numpy().T
    post_18 = post_18_df.to_numpy().T
    post_28 = post_28_df.to_numpy().T
    post_77 = post_77_df.to_numpy().T

    baseline_minus_63_control = baseline_minus_63_df_control.to_numpy().T
    baseline_minus_2_control = baseline_minus_2_df_control.to_numpy().T
    baseline_minus_1_control = baseline_minus_1_df_control.to_numpy().T
    baseline_control = baseline_df_control.to_numpy().T
    baseline_full = baseline_full_df.to_numpy().T
    abx_1_control = abx_1_df_control.to_numpy().T
    abx_2_control = abx_2_df_control.to_numpy().T
    abx_3_control = abx_3_df_control.to_numpy().T
    abx_4_control = abx_4_df_control.to_numpy().T
    abx_5_control = abx_5_df_control.to_numpy().T
    post_6_control = post_6_df_control.to_numpy().T
    post_7_control = post_7_df_control.to_numpy().T
    post_8_control = post_8_df_control.to_numpy().T
    post_10_control = post_10_df_control.to_numpy().T
    post_18_control = post_18_df_control.to_numpy().T
    post_28_control = post_28_df_control.to_numpy().T
    post_77_control = post_77_df_control.to_numpy().T

    filtered_keys = [sample_id for sample_id, _ in treatment_subjects]
    filtered_keys_control = keys_control

    times_post_abx = [6, 7, 8, 10, 18, 28, 77]

    return {
        "abundance_table": df,
        "baseline_df": baseline_df,
        "baseline_minus_63_df": baseline_minus_63_df,
        "baseline_minus_2_df": baseline_minus_2_df,
        "baseline_minus_1_df": baseline_minus_2_df,
        "post_abx_df": post_77_df,
        "post_abx_6_df": post_6_df,
        "post_abx_7_df": post_7_df,
        "post_abx_8_df": post_8_df,
        "post_abx_10_df": post_10_df,
        "post_abx_18_df": post_18_df,
        "post_abx_28_df": post_28_df,
        "abx_df": abx_5_df,
        "abx_4_df": abx_4_df,
        "abx_3_df": abx_3_df,
        "abx_2_df": abx_2_df,
        "abx_1_df": abx_1_df,
        "baseline_full_df": baseline_full_df,

        "baseline_df_control": baseline_df_control,
        "baseline_minus_63_df_control": baseline_minus_63_df_control,
        "baseline_minus_2_df_control": baseline_minus_2_df_control,
        "baseline_minus_1_df_control": baseline_minus_1_df_control,
        "post_abx_df_control": post_77_df_control,
        "post_abx_6_df_control": post_6_df_control,
        "post_abx_7_df_control": post_7_df_control,
        "post_abx_8_df_control": post_8_df_control,
        "post_abx_10_df_control": post_10_df_control,
        "post_abx_18_df_control": post_18_df_control,
        "post_abx_28_df_control": post_28_df_control,
        "abx_df_control": abx_5_df_control,
        "abx_4_df_control": abx_4_df_control,
        "abx_3_df_control": abx_3_df_control,
        "abx_2_df_control": abx_2_df_control,
        "abx_1_df_control": abx_1_df_control,

        "baseline": baseline,
        "baseline_minus_63": baseline_minus_63,
        "baseline_minus_2": baseline_minus_2,
        "baseline_minus_1": baseline_minus_1,
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
        "baseline_full": baseline_full,

        "baseline_control": baseline_control,
        "baseline_minus_63_control": baseline_minus_63_control,
        "baseline_minus_2_control": baseline_minus_2_control,
        "baseline_minus_1_control": baseline_minus_1_control,
        "post_abx_control": post_77_control,
        "post_abx_6_control": post_6_control,
        "post_abx_7_control": post_7_control,
        "post_abx_8_control": post_8_control,
        "post_abx_10_control": post_10_control,
        "post_abx_18_control": post_18_control,
        "post_abx_28_control": post_28_control,
        "abx_control": abx_5_control,
        "abx_4_control": abx_4_control,
        "abx_3_control": abx_3_control,
        "abx_2_control": abx_2_control,
        "abx_1_control": abx_1_control,

        "post_abx_cohorts": [
            post_6,
            post_7,
            post_8,
            post_10,
            post_18,
            post_28,
            post_77
        ],
        "post_abx_cohorts_df": [
            post_6_df,
            post_7_df,
            post_8_df,
            post_10_df,
            post_18_df,
            post_28_df,
            post_77_df
        ],
        "post_abx_cohorts_control": [
            post_6_control,
            post_7_control,
            post_8_control,
            post_10_control,
            post_18_control,
            post_28_control,
            post_77_control
        ],
        "post_abx_cohorts_df_control": [
            post_6_df_control,
            post_7_df_control,
            post_8_df_control,
            post_10_df_control,
            post_18_df_control,
            post_28_df_control,
            post_77_df_control
        ],

        "times_post_abx": times_post_abx,

        "keys": keys,
        "keys_control": keys_control,
        "filtered_keys": filtered_keys,
        "filtered_keys_control": filtered_keys_control
    }
