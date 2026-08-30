from src.host_specific_recovery.simulations.historical_contingency import HC
from utils.general_functions import calc_similarity_standard
import numpy as np
import random


def run_HC_simulation(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                      final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric, alpha, method,
                      multiprocess, n_jobs, numpy_seed, random_seed):

    # define random seeds
    np.random.seed(numpy_seed)
    random.seed(random_seed)

    # No switch off
    HC_object = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                   final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
                   alpha, method, multiprocess, False, False, n_jobs)
    results = HC_object.get_results()

    base_sim = results["Y_0"]
    abx_sim = results["Y_p"]
    post_sim = results["y_s"]
    post_sim_others = results["Y_s"]

    # Switch off
    np.random.seed(numpy_seed)
    random.seed(random_seed)
    HC_object_off = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                       final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
                       alpha, method, multiprocess, True, True, n_jobs)
    results_off = HC_object_off.get_results()

    base_sim_off = results_off["Y_0"]
    abx_sim_off = results_off["Y_p"]
    post_sim_off = results_off["y_s"]
    post_sim_others_off = results_off["Y_s"]

    sims_new, sims_survived = calc_similarity_standard(post_sim, abx_sim, post_sim_others)
    sizes = []
    for smp in post_sim_others:
        sizes.append(np.size(np.nonzero(smp)))
    sizes = np.array(sizes)
    sims_new_off, sims_survived_off = calc_similarity_standard(post_sim_off, abx_sim_off, post_sim_others_off)
    sizes_off = []
    for smp in post_sim_others_off:
        sizes_off.append(np.size(np.nonzero(smp)))
    sizes_off = np.array(sizes_off)

    return {
        "base_sim": base_sim,
        "abx_sim": abx_sim,
        "post_sim": post_sim,
        "post_sim_others": post_sim_others,
        "sims_new": sims_new,
        "sims_survived": sims_survived,
        "sizes": sizes,
        "base_sim_off": base_sim_off,
        "abx_sim_off": abx_sim_off,
        "post_sim_off": post_sim_off,
        "post_sim_others_off": post_sim_others_off,
        "sims_new_off": sims_new_off,
        "sims_survived_off": sims_survived_off,
        "sizes_off": sizes_off,
    }
