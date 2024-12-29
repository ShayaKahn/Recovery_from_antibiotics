import numpy as np
import random
from methods.historical_contingency  import HC

# define random seeds
np.random.seed(10)
random.seed(42)

# parameters
num_samples = 100
pool_size = 200
num_survived_min = 100
num_survived_max = 101
sigma = 10
mean = 0
c = 0.05
delta = 1e-5
final_time = 1000
max_step = 0.05
epsilon = 1e-4
threshold = 1e-3
min_growth = 0.99
max_growth = 1
symmetric = False
alpha = None
method = 'RK45'
multiprocess = True

# No switch off
HC_object = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
               final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
               alpha, method, multiprocess, False)
results = HC_object.get_results()

post_sim = results["filtered_post_perturbed_state"]
post_sim_others = results["filtered_post_perturbed_state_others"]
ABX_sim = results["perturbed_state"]

np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim.npy", post_sim)
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_others.npy", post_sim_others)
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ABX_sim.npy", ABX_sim)

HC_object = HC(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
               final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
               alpha, method, multiprocess, True)
results = HC_object.get_results()

post_sim = results["filtered_post_perturbed_state"]
post_sim_others = results["filtered_post_perturbed_state_others"]
ABX_sim = results["perturbed_state"]

np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_off.npy", post_sim)
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/post_sim_others_off.npy", post_sim_others)
np.save("C:/Users/USER/OneDrive/Desktop/Antibiotics/Results/ABX_sim_off.npy", ABX_sim)
