from src.host_specific_recovery.io.Sewunet_et_al_loader import load_Sewunet_et_al_data
from src.host_specific_recovery.analysis.survived_species_analysis import run_survived_species_analysis
from src.host_specific_recovery.visualizations.plot_survived_species_analysis import plot_survived_species_analysis

dataset = load_Sewunet_et_al_data()
outputs = run_survived_species_analysis(dataset)
plot_survived_species_analysis = plot_survived_species_analysis(outputs,
                                                                x_vals=[0, 10, 14, 21, 90, 180],
                                                                x_labels=["Baseline", "ABX", "Day 4", "Day 11",
                                                                          "Day 80", "Day 180"])
