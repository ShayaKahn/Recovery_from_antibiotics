from src.host_specific_recovery.analysis.GLV_similarity_correlation import run_HC_simulation
from src.host_specific_recovery.io.writers import write_hc
import numpy as np
import random

np.random.seed(48)
random.seed(98)

# parameters
num_samples = 100
pool_size = 100
num_survived_min = 25
num_survived_max = 25
sigma = 5
mean = 0
c = 0.075
delta = 1e-5
final_time = 1000
max_step = 0.05
epsilon = 1e-4
threshold = 1e-4
min_growth = 1
max_growth = 1
symmetric = False
alpha = None
method = 'RK45'
multiprocess = True
n_jobs = 6
numpy_seed = 48
random_seed = 98

outputs = run_HC_simulation(num_samples, pool_size, num_survived_min, num_survived_max, mean, sigma, c, delta,
                            final_time, max_step, epsilon, threshold, min_growth, max_growth, symmetric,
                            alpha, method, multiprocess, n_jobs, numpy_seed, random_seed)

write_hc(outputs,"C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                 "Recovery_from_antibiotics/results/GLV_simulation/")
print(0)
