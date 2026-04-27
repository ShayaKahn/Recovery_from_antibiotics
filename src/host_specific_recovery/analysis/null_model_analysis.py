from src.host_specific_recovery.statistical_models.null_model import NullModel
import numpy as np

def run_null_model_analysis(dataset: dict, timepoints_val:int,  method: str = "Specificity",
                            num_reals: int = 10000) -> dict:

    # initialize results
    real_dist_container = []
    shuffled_dist_container = []

    # iterate over the test subjects
    for j, (base, abx) in enumerate(zip(dataset["baseline"], dataset["abx"])):
        post_abx_matrix = np.vstack([post[j, :] for post in dataset["post_abx_cohorts"]])
        # apply null model
        null_model_obj = NullModel(base, abx, dataset["baseline_full"], post_abx_matrix, num_reals, timepoints_val)
        dist_real, dist_shuffled = null_model_obj.distance(method)
        real_dist_container.append(dist_real)
        shuffled_dist_container.append(dist_shuffled)
    # Convert to similarities
    real_sim_container = [1 - r for r in real_dist_container]
    shuffled_sim_container = [1 - sp for sp in shuffled_dist_container]

    return {"real_dist": real_sim_container, "shuffled_dist": shuffled_sim_container,
            "filtered_keys": dataset["filtered_keys"]}
