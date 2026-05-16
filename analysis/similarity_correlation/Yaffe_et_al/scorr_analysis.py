from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.io.writers import save_sc_results
from src.host_specific_recovery.analysis.similarity_correlation import run_similarity_correlation

dataset = load_Yaffe_et_al_data()
outputs = run_similarity_correlation(dataset, timepoints_val=4)
save_sc_results(outputs, "C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                         "Recovery_from_antibiotics/results/Yaffe_et_al/similarity_correlation")
