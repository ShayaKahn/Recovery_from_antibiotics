import numpy as np
from src.host_specific_recovery.metrics.similarity import Similarity

def calc_similarity_standard(steady_state, ABX, steady_state_mat):
    survived = np.where((steady_state != 0) & (ABX[0, :] != 0))[0]
    new = np.where((steady_state != 0) & (ABX[0, :] == 0))[0]
    sims_new = []
    sims_survived = []
    for j, abx in enumerate(steady_state_mat):
        sim_new = Similarity(abx[new], steady_state[new], method='Jaccard').calculate_similarity()
        sim_survived = Similarity(abx[survived], steady_state[survived], method='Jaccard').calculate_similarity()
        sims_new.append(sim_new)
        sims_survived.append(sim_survived)
    return sims_new, sims_survived
