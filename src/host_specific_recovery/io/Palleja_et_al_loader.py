from src.host_specific_recovery.io.common import *
from typing import Dict, List

rel_abund_rarefied = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\Antibiotics\Recovery\Data\annotated.mOTU.rel_abund.rarefied.tsv",
                                 sep='\t')

def normalize_cohort(cohort):
    # normalization function
    if cohort.ndim == 1:
        cohort_normalized = cohort / cohort.sum()
    else:
        cohort_normalized = cohort / np.linalg.norm(cohort, ord=1, axis=1, keepdims=True)
    return cohort_normalized

def _build_keys() -> List[str]:
    return [f'Subject {j+1}' for j in range(12)]

def _build_filtered_keys() -> List[str]:
    return ['Subject 2', 'Subject 3', 'Subject 4', 'Subject 5', 'Subject 6', 'Subject 7', 'Subject 9', 'Subject 11',
            'Subject 12']

def _build_baseline_columns() -> List[str]:
    return ['ERAS1_Dag0', 'ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0', 'ERAS6_Dag0', 'ERAS7_Dag0',
            'ERAS8_Dag0', 'ERAS9_Dag0', 'ERAS10_Dag0', 'ERAS11_Dag0', 'ERAS12_Dag0']

def _baseline_columns_appear_4() -> List[str]:
    return ['ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0', 'ERAS6_Dag0', 'ERAS7_Dag0', 'ERAS9_Dag0',
            'ERAS11_Dag0', 'ERAS12_Dag0']

def _build_columns_4() -> List[str]:
    return ['ERAS2_Dag4opt', 'ERAS3_Dag4', 'ERAS4_Dag4opt', 'ERAS5_Dag4', 'ERAS6_Dag4opt', 'ERAS7_Dag4opt',
            'ERAS9_Dag4', 'ERAS11_Dag4opt', 'ERAS12_Dag4opt']

def _build_columns_8() -> List[str]:
    return ['ERAS1_Dag8',  'ERAS2_Dag8', 'ERAS3_Dag8', 'ERAS4_Dag8opt', 'ERAS5_Dag8', 'ERAS6_Dag8opt', 'ERAS7_Dag8',
             'ERAS8_Dag8', 'ERAS9_Dag8', 'ERAS10_Dag8', 'ERAS11_Dag8', 'ERAS12_Dag8']

def _build_columns_8_appear_4() -> List[str]:
    return ['ERAS2_Dag8', 'ERAS3_Dag8', 'ERAS4_Dag8opt', 'ERAS5_Dag8', 'ERAS6_Dag8opt', 'ERAS7_Dag8', 'ERAS9_Dag8',
            'ERAS11_Dag8', 'ERAS12_Dag8']

def _build_columns_42() -> List[str]:
    return ['ERAS1_Dag42', 'ERAS2_Dag42', 'ERAS3_Dag42', 'ERAS4_Dag42', 'ERAS5_Dag42', 'ERAS6_Dag42', 'ERAS7_Dag42',
            'ERAS8_Dag42', 'ERAS9_Dag42', 'ERAS10_Dag42', 'ERAS11_Dag42', 'ERAS12_Dag42']

def _build_columns_42_appear_4() -> List[str]:
    return ['ERAS2_Dag42', 'ERAS3_Dag42', 'ERAS4_Dag42', 'ERAS5_Dag42', 'ERAS6_Dag42', 'ERAS7_Dag42', 'ERAS9_Dag42',
            'ERAS11_Dag42', 'ERAS12_Dag42']

def _build_columns_180() -> List[str]:
    return ['ERAS1_Dag180', 'ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180',
            'ERAS7_Dag180', 'ERAS8_Dag180', 'ERAS9_Dag180', 'ERAS10_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

def _build_columns_180_appear_4() -> List[str]:
    return ['ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180', 'ERAS7_Dag180',
            'ERAS9_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

def load_Pallega_et_al_data() -> Dict[str, Optional[object]]:

    # load data
    rel_abund_rarefied = load_csv_df(
        r"C:\Users\USER\OneDrive\Desktop\Antibiotics\Recovery\Data\annotated.mOTU.rel_abund.rarefied.tsv",
        sep='\t')
    baseline_columns = _build_baseline_columns()
    columns_4 = _build_columns_4()
    columns_8 = _build_columns_8()
    columns_8_appear_4 = _build_columns_8_appear_4()
    columns_42 = _build_columns_42()
    columns_42_appear_4 = _build_columns_42_appear_4()
    columns_180 = _build_columns_180()
    columns_180_appear_4 = _build_columns_180_appear_4()

    baseline_full = transpose_numeric(rel_abund_rarefied[baseline_columns], norm=False)
    baseline_filtered = transpose_numeric(rel_abund_rarefied[baseline_columns], norm=True)
    ABX_filtered = transpose_numeric(rel_abund_rarefied[columns_4], norm=True)
    post_ABX_8_filtered = transpose_numeric(rel_abund_rarefied[columns_8], norm=True)
    post_ABX_8_appear_4_filtered = transpose_numeric(rel_abund_rarefied[columns_8_appear_4], norm=True)
    post_ABX_42_filtered = transpose_numeric(rel_abund_rarefied[columns_42], norm=True)
    post_ABX_42_appear_4_filtered = transpose_numeric(rel_abund_rarefied[columns_42_appear_4], norm=True)
    post_ABX_filtered = transpose_numeric(rel_abund_rarefied[columns_180], norm=True)
    post_ABX_appear_4_filtered = transpose_numeric(rel_abund_rarefied[columns_180_appear_4], norm=True)

    keys = _build_keys()
    filtered_keys = _build_filtered_keys()

    keys = list_to_numpy(keys)
    filtered_keys = list_to_numpy(filtered_keys)

    return {
        "baseline_full": baseline_full,
        "baseline": baseline_filtered,
        "post_abx": post_ABX_filtered,
        "post_abx_8": post_ABX_8_filtered,
        "post_abx_8_appear_4": post_ABX_8_appear_4_filtered,
        "post_abx_42": post_ABX_42_filtered,
        "post_abx_42_appear_4": post_ABX_42_appear_4_filtered,
        "abx": ABX_filtered,
        "abx_4": post_ABX_appear_4_filtered,

        "post_abx_cohorts": [
            post_ABX_8_appear_4_filtered,
            post_ABX_42_appear_4_filtered,
            post_ABX_appear_4_filtered
        ],

        "keys": keys,
        "filtered_keys": filtered_keys
    }
