from src.host_specific_recovery.simulations.historical_contingency import HC
import numpy as np
import random

def run_HC_simulation(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                      final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric, alpha, method,
                      multiprocess, switch_off, n_jobs, numpy_seed, random_seed):

    # define random seeds
    np.random.seed(numpy_seed)
    random.seed(random_seed)

    # No switch off
    HC_object = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                   final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
                   alpha, method, multiprocess, switch_off, n_jobs)
    results = HC_object.get_results()

    base_sim = results["Y_0"]
    abx_sim = results["Y_p"]
    post_sim = results["y_s"]
    post_sim_others = results["Y_s"]

    # Switch off
    HC_object_off = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                       final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
                       alpha, method, multiprocess, True)
    results_off = HC_object_off.get_results()

    base_sim_off = results["Y_0"]
    abx_sim_off = results["Y_p"]
    post_sim_off = results["y_s"]
    post_sim_others_off = results["Y_s"]

    return {
        "base_sim": base_sim,
        "abx_sim": abx_sim,
        "post_sim": post_sim,
        "post_sim_others": post_sim_others,
        "base_sim_off": base_sim_off,
        "abx_sim_off": abx_sim_off,
        "post_sim_off": post_sim_off,
        "post_sim_others_off": post_sim_others_off
    }
