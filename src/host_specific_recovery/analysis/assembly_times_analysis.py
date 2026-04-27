from src.host_specific_recovery.utils.general_utils import subset
import numpy as np
import pandas as pd
import copy

def run_assembly_times_analysis(dataset: dict, timepoint_threshold:int, strict: bool = True) -> dict:

    post_ABX = dataset["post_abx_cohorts"]

    n = len(dataset["post_abx_cohorts"])
    cat = [f'timepoint_{j + 1}' for j in range(n)]

    filtered_keys = dataset["filtered_keys"]
    baseline = dataset["baseline"]
    ABX = dataset["abx"]

    percentage_returned_lst = []
    percentage_new_lst = []
    new_counts = []
    returned_counts = []
    contingency_tables = {}

    for i, key, abx, base in enumerate(zip(filtered_keys, ABX, baseline)):

        post_matrix = np.vstack([p[i, :] for p in post_ABX])

        new_species = subset(post_matrix, base, abx, strict, new=True)
        new_count = [n.sum() for n in new_species]
        new_counts.append(new_count)

        returned_species = subset(post_matrix, base, abx, strict, new=False)
        returned_count = [r.sum() for r in returned_species]
        returned_counts.append(returned_count)

        contingency_table = pd.DataFrame({
            'New': new_count,
            'Returned': returned_count},
            index=cat)
        contingency_tables[key] = contingency_table
        contingency_table_copy = copy.deepcopy(contingency_table)
        contingency_table_copy.loc["Early"] = contingency_table_copy.iloc[:timepoint_threshold].sum()
        contingency_table_copy.loc["Late"] = contingency_table_copy.iloc[timepoint_threshold:].sum()
        percentage_returned = contingency_table_copy["Returned"].loc["Late"] / (
                contingency_table_copy["Returned"].loc["Late"] + contingency_table_copy["Returned"].loc["Early"]) * 100
        percentage_returned_lst.append(percentage_returned)
        percentage_new = contingency_table_copy["New"].loc["Late"] / (
                    contingency_table_copy["New"].loc["Late"] + contingency_table_copy["New"].loc["Early"]) * 100
        percentage_new_lst.append(percentage_new)

    percentage_returned_lst = np.array(percentage_returned_lst)
    percentage_new_lst = np.array(percentage_new_lst)

    return {
        "percentage_returned_lst": percentage_returned_lst,
        "percentage_new_lst": percentage_new_lst,
        "new_counts": new_counts,
        "returned_counts": returned_counts,
        "contingency_tables": contingency_tables
    }
