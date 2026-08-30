from src.host_specific_recovery.analysis.CR_similarity_correlation import run_HC_CR_simulation
from src.host_specific_recovery.io.writers import write_hc
import numpy as np
import random

np.random.seed(48)
random.seed(98)

num_samples = 500

pool_size = 40#32#64
n_resources = 60#48#96
num_survived_min = 20#8#8#16#25
num_survived_max = 20#int(16)#8#8#16#25

delta = 0.6e-3#1e-5
final_time = 25000#20000#50000
max_step = 0.5#0.05
immigration_abundance = 1e-10#1e-8

threshold = 1e-4

resource_min = 1.0
resource_max = 2.5
min_rho = 1.5
max_rho = 2.0
min_mortality = 0.1
max_mortality = 0.3
consumer_efficiency = 0.6

growth_mean = 1.0

upsilon = 0.35#0.7#0.85

matrix_sampling = "iid"#"general"
sample_fixed_point = True

filter_by_event = True#False
method = "RK45"
multiprocess = False#True
n_jobs = 8
numpy_seed = 48
random_seed = 98

outputs = run_HC_CR_simulation(num_samples, pool_size, n_resources, num_survived_min, num_survived_max,
                               delta, final_time, max_step, immigration_abundance, threshold, resource_min,
                               resource_max, min_rho, max_rho, min_mortality, max_mortality, consumer_efficiency,
                               method, multiprocess, n_jobs, numpy_seed, random_seed, growth_mean,
                               filter_by_event, upsilon, matrix_sampling, sample_fixed_point)

write_hc(outputs, "C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                  "Recovery_from_antibiotics/results/CR_simulation/", model_name="cr")

print(0)
