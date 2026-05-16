from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.survived_species_analysis import run_survived_species_analysis
from src.host_specific_recovery.visualizations.plot_survived_species_analysis import plot_survived_species_analysis

dataset = load_Messaoudene_et_al_data()
outputs = run_survived_species_analysis(dataset)
plot_survived_species_analysis = plot_survived_species_analysis(outputs,
                                                                x_vals=[0, 10, 14, 17, 22, 30, 42],
                                                                x_labels=["Baseline", "ABX", "Day 9", "Day 12",
                                                                          "Day 16", "Day 25", "Day 37"])
