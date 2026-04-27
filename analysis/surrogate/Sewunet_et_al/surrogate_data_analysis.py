"""
import os
import numpy as np

os.chdir('C:/Users/USER/OneDrive/Desktop/Antibiotics')
from Classes.surrogate import Surrogate
from general_functions import *
from Analysis.dataset_2_load_data import *

#### Surrogate data analysis ####

# initialize results
results_jaccard = []
results_obj = []
results_jaccard_right = []
results_obj_right = []
results_jaccard_mid = []
results_obj_mid = []
results_jaccard_naive = []
results_obj_naive = []

method = "jensenshannon"

# iterate over the test subjects - apply the three methods: Naive, mid (remove s), and remove w
for i, key in enumerate(filtered_keys):
    # create and define inputs
    subject_base = {key: baseline[i, :]}
    subject_base_dict = create_cohort_dict(keys, key, baseline_full)
    subject_test = post_ABX[i, :]
    subject_test_90 = post_ABX_90[i, :]
    subject_test_21 = post_ABX_21[i, :]
    subject_test_14 = post_ABX_14[i, :]
    subject_ABX = ABX[i, :]
    test_post_ABX_matrix = np.vstack([subject_test_14, subject_test_21, subject_test_90, subject_test])
    # apply surrogate data analysis
    results_subject_obj = Surrogate(subject_base_dict, subject_base, test_post_ABX_matrix, subject_ABX, timepoints=2,
                                    strict=False)
    results_subject_jaccard = results_subject_obj.apply_surrogate_data_analysis(method=method)
    results_jaccard.append(results_subject_jaccard)
    results_obj.append(results_subject_obj)
    ## SDA - Right condition
    #results_subject_obj_right = Surrogate(subject_base_dict, subject_base, test_post_ABX_matrix, subject_ABX,
    #                                      timepoints=2, strict=False, left_cond=False)
    #results_subject_jaccard_right = results_subject_obj_right.apply_surrogate_data_analysis(method=method)
    #results_jaccard_right.append(results_subject_jaccard_right)
    #results_obj_right.append(results_subject_obj_right)
    # Remove only survived species
    results_subject_obj_mid = Surrogate(subject_base_dict, subject_base, test_post_ABX_matrix, subject_ABX,
                                        timepoints=0, strict=False)
    results_subject_jaccard_mid = results_subject_obj_mid.apply_surrogate_data_analysis(method=method)
    results_jaccard_mid.append(results_subject_jaccard_mid)
    results_obj_mid.append(results_subject_obj_mid)
    # Naive method
    results_subject_obj_naive = Surrogate(subject_base_dict, subject_base, test_post_ABX_matrix, subject_ABX,
                                          timepoints=0, strict=False, naive=True)
    results_subject_jaccard_naive = results_subject_obj_naive.apply_surrogate_data_analysis(method=method)
    results_jaccard_naive.append(results_subject_jaccard_naive)
    results_obj_naive.append(results_subject_obj_naive)

j = [res[key] for res, key in zip(results_jaccard, filtered_keys)]

j_others = []
for specific_key, res_j in zip(filtered_keys, results_jaccard):
    j_others.append([res_j[key] for key in keys if key != specific_key])

#j_right = [res[key] for res, key in zip(results_jaccard_right, filtered_keys)]

#j_others_right = []
#for specific_key, res_j in zip(filtered_keys, results_jaccard_right):
#    j_others_right.append([res_j[key] for key in keys if key != specific_key])

j_mid = [res[key] for res, key in zip(results_jaccard_mid, filtered_keys)]

j_others_mid = []
for specific_key, res_j in zip(filtered_keys, results_jaccard_mid):
    j_others_mid.append([res_j[key] for key in keys if key != specific_key])

j_naive = [res[key] for res, key in zip(results_jaccard_naive, filtered_keys)]

j_others_naive = []
for specific_key, res_j in zip(filtered_keys, results_jaccard_naive):
    j_others_naive.append([res_j[key] for key in keys if key != specific_key])

# calculate the ranks
ranks = np.array([len(keys) - np.sum((j_val > j_val_others)) for j_val, j_val_others in zip(j, j_others)])
#ranks_right = np.array([len(keys) - np.sum((j_val > j_val_others)) for j_val, j_val_others in zip(j_right,
##                                                                                                    j_others_right)])
ranks_mid = np.array([len(keys) - np.sum((j_val > j_val_others)) for j_val, j_val_others in zip(j_mid, j_others_mid)])
ranks_naive = np.array([len(keys) - np.sum((j_val > j_val_others)) for j_val, j_val_others in zip(j_naive,
                                                                                                  j_others_naive)])

np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ranks_mid_dataset_2.npy", np.array(ranks_mid))
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ranks_dataset_2.npy", np.array(ranks))
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ranks_naive_dataset_2.npy", np.array(ranks_naive))
#np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ranks_right_dataset_2.npy", np.array(ranks_right))

if __name__ == "__main__":

    # Save results
    similarities = []
    for d in results_jaccard:
        sims = list(d.values())
        for s in sims:
            similarities.append(s)

    np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarities_OTU.npy", similarities)

    #similarities_right = []
    #for d in results_jaccard_right:
    #    sims = list(d.values())
    #    for s in sims:
    #        similarities_right.append(s)

    #np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarities_OTU_right.npy", similarities_right)

    similarities_mid = []
    for d in results_jaccard_mid:
        sims = list(d.values())
        for s in sims:
            similarities_mid.append(s)

    np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarities_OTU_mid.npy", similarities)

    similarities_naive = []
    for d in results_jaccard_naive:
        sims = list(d.values())
        for s in sims:
            similarities_naive.append(s)

    np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/similarities_OTU_naive.npy", similarities)
"""