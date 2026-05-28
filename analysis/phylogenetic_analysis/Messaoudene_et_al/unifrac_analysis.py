from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.phylogenetic_analysis import run_phylogenetic_test
from src.host_specific_recovery.visualizations.plot_phylogenetic_analysis import plot_unifrac_similarity_scatter
from src.host_specific_recovery.io.writers import save_phylogenetic_test_results

dataset = load_Messaoudene_et_al_data()
outputs = run_phylogenetic_test(dataset)

path = ("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results/Messaoudene_et_al/"
        "phylogenetic_analysis")

save_phylogenetic_test_results(outputs, path)
plot_unifrac_similarity_scatter(outputs, path=path)

print(0)
