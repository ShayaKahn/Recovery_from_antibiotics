from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.analysis.cross_probability_analysis import (run_cross_species_probability_analysis,
                                                                            run_cross_subject_probability_analysis,
                                                                    run_cross_subject_probability_analysis_abundance)
from src.host_specific_recovery.visualizations.plot_cross_probability_analysis import (plot_cross_species,
                                                                                plot_colonization_probabilities_bubble,
                                                                                scatter_with_identity)
import matplotlib.pyplot as plt

dataset = load_Yaffe_et_al_data()
outputs_cross_species = run_cross_species_probability_analysis(dataset, min_estimate=5)
plot_cross_species(outputs_cross_species, title='', color='#4A6FA5',
                   path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results/"
                        "Yaffe_et_al/cross_probability_analysis/cross_species")

outputs_cross_subject = run_cross_subject_probability_analysis(dataset)
plot_colonization_probabilities_bubble(outputs_cross_subject, color='#4A6FA5',
                                       path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                                            "Recovery_from_antibiotics/results/Yaffe_et_al/cross_probability_analysis/"
                                            "cross_subject")

outputs_abundance = run_cross_subject_probability_analysis_abundance(outputs_cross_subject)

scatter_with_identity(outputs_abundance["new_taxa_abundances"][:, 0],
                      outputs_abundance["new_taxa_abundances"][:, 1],
                      color="#DC3912",
                      path = "C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/"
                             "results/Yaffe_et_al/cross_probability_analysis/new_abun")

scatter_with_identity(outputs_abundance["ret_taxa_abundances"][:, 0],
                      outputs_abundance["ret_taxa_abundances"][:, 1],
                      color="#3366CC",
                      path = "C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/"
                             "results/Yaffe_et_al/cross_probability_analysis/ret_abun")

