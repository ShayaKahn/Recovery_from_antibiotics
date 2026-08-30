import random
import numpy as np
from src.host_specific_recovery.simulations.historical_contingency_CR import HC_CR
from utils.general_functions import calc_similarity_standard


def run_HC_CR_simulation(num_samples, pool_size, n_resources, num_survived_min, num_survived_max,
                         delta, final_time, max_step, immigration_abundance, threshold, resource_min=0.5,
                         resource_max=2.0, min_rho=1.0, max_rho=1.0, min_mortality=0.05, max_mortality=0.3,
                         consumer_efficiency=0.5, method='RK45', multiprocess=True, n_jobs=4,
                         numpy_seed=None, random_seed=None, growth_mean=1.0, filter_by_event=True,
                         upsilon=0.4, matrix_sampling="general", sample_fixed_point=True):
    """Run historical contingency with the Liu-style consumer-resource model.

    This is the consumer-resource analogue of GLV_similarity_correlation.run_HC_simulation.
    It returns the same species-level keys used by the GLV wrapper, plus resource-level
    keys produced by HC_CR.
    """

    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if random_seed is not None:
        random.seed(random_seed)

    HC_object = HC_CR(
        num_samples=num_samples,
        pool_size=pool_size,
        n_resources=n_resources,
        num_survived_min=num_survived_min,
        num_survived_max=num_survived_max,
        delta=delta,
        final_time=final_time,
        max_step=max_step,
        immigration_abundance=immigration_abundance,
        eta=threshold,
        resource_min=resource_min,
        resource_max=resource_max,
        min_rho=min_rho,
        max_rho=max_rho,
        min_mortality=min_mortality,
        max_mortality=max_mortality,
        consumer_efficiency=consumer_efficiency,
        method=method,
        multiprocess=multiprocess,
        n_jobs=n_jobs,
        filter_by_event=filter_by_event,
        upsilon=upsilon,
        matrix_sampling=matrix_sampling,
        sample_fixed_point=sample_fixed_point,
    )
    results = HC_object.get_results()

    resource_base_sim = results["R_0"]
    base_sim = results["Y_0"]
    resource_abx_sim = results["R_p"]
    abx_sim = results["Y_p"]
    resource_post_sim = results["r_s"]
    post_sim = results["y_s"]
    resource_post_sim_others = results["R_s"]
    post_sim_others = results["Y_s"]
    C = results["C"]
    G = results["G"]
    rho = results["rho"]
    mu = results["mu"]
    R_star = results["R_star"]
    S_star = results["S_star"]
    upsilon = results["upsilon"]
    matrix_sampling = results["matrix_sampling"]
    sample_fixed_point = results["sample_fixed_point"]
    empirical_growth_consumption_correlation = results["empirical_growth_consumption_correlation"]

    sims_new, sims_survived = calc_similarity_standard(post_sim, abx_sim, post_sim_others)
    sizes = np.array([np.size(np.nonzero(smp)) for smp in post_sim_others])

    return {
        "resource_base_sim": resource_base_sim,
        "base_sim": base_sim,
        "resource_abx_sim": resource_abx_sim,
        "abx_sim": abx_sim,
        "resource_post_sim": resource_post_sim,
        "post_sim": post_sim,
        "resource_post_sim_others": resource_post_sim_others,
        "post_sim_others": post_sim_others,
        "C": C,
        "G": G,
        "rho": rho,
        "mu": mu,
        "R_star": R_star,
        "S_star": S_star,
        "upsilon": upsilon,
        "matrix_sampling": matrix_sampling,
        "sample_fixed_point": sample_fixed_point,
        "empirical_growth_consumption_correlation": empirical_growth_consumption_correlation,
        "sims_new": sims_new,
        "sims_survived": sims_survived,
        "sizes": sizes,
    }

run_HC_simulation = run_HC_CR_simulation
