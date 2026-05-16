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

def calculate_characteristic_time(dataset):
    """
    This function calculates the characteristic times for returned and new species for each subject.
    :param dataset: dictionary containing the dataset information/
    :return: returned_characteristic_time_dict, new_characteristic_time_dict
    """

    times = np.insert(np.array(dataset["times_post_abx"]), 0, 0)

    def characteristic_time(prop_array, times):
        average_times = np.array([(times[i] + times[i + 1]) / 2 for i in range(len(times) - 1)])
        return np.dot(prop_array, average_times)

    baseline_cohort = dataset["baseline"]
    ABX_cohort = dataset["abx"]
    ids = dataset["filtered_keys"]
    post_ABX_tensor = np.stack(dataset["post_abx_cohorts"], axis=1)
    returned_characteristic_time_dict = {}
    new_characteristic_time_dict = {}
    valid_ids = []
    for i in range(len(ids)):
        new_species = subset(post_ABX_tensor[i, ...], baseline_cohort[i, :], ABX_cohort[i, :], True, new=True)
        new_counts = np.array(new_species).sum(axis=1)
        returned_species = subset(post_ABX_tensor[i, ...], baseline_cohort[i, :], ABX_cohort[i, :], True, new=False)
        returned_counts = np.array(returned_species).sum(axis=1)
        new_total = np.sum(new_counts)
        returned_total = np.sum(returned_counts)
        if new_total == 0 or returned_total == 0:
            continue

        new_props = new_counts / new_total
        returned_props = returned_counts / returned_total
        returned_characteristic_time = characteristic_time(returned_props.reshape(1, -1), times)
        new_characteristic_time = characteristic_time(new_props.reshape(1, -1), times)
        if not np.all(np.isfinite(returned_characteristic_time)) or not np.all(np.isfinite(new_characteristic_time)):
            continue

        returned_characteristic_time_dict[ids[i]] = returned_characteristic_time
        new_characteristic_time_dict[ids[i]] = new_characteristic_time
        valid_ids.append(ids[i])
    return {
        "returned_characteristic_time_dict": returned_characteristic_time_dict,
        "new_characteristic_time_dict": new_characteristic_time_dict,
        "keys": valid_ids
    }
