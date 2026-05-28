from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.analysis.survived_species_analysis import run_survived_species_analysis
from src.host_specific_recovery.visualizations.plot_survived_species_analysis import plot_survived_species_analysis

dataset = load_Yaffe_et_al_data()
outputs = run_survived_species_analysis(dataset)
plot_survived_species_analysis = plot_survived_species_analysis(outputs,
                                                                x_vals=[0, 10, 11, 12, 13, 15, 23, 33, 82],
                                                                x_labels=["Baseline", "ABX", ".", ".", ".",
                                                                          "Day 10", "Day 18", "Day 28", "Day 77"],
                                                                dir="C:/Users/USER/OneDrive/Desktop/"
                                                                    "Recovery_from_antibiotics/"
                                                                    "Recovery_from_antibiotics/results/Yaffe_et_al/"
                                                                    "survived_species_analysis/survived_plot",
                                                                xticks_fontsize=50, ylable_fontsize=100,
                                                                ytick_fontsize=70)
