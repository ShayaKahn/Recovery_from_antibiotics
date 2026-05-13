from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.cross_probability_analysis import (run_cross_species_probability_analysis,
                                                                            run_cross_subject_probability_analysis)
from src.host_specific_recovery.visualizations.plot_cross_probability_analysis import (plot_cross_species,
                                                                                plot_colonization_probabilities_bubble)

dataset = load_Messaoudene_et_al_data()
outputs_cross_species = run_cross_species_probability_analysis(dataset, min_estimate=5)
plot_cross_species(outputs_cross_species, title='', color='#1A71B8')

outputs_cross_subject = run_cross_subject_probability_analysis(dataset)
plot_colonization_probabilities_bubble(outputs_cross_subject, color='#1A71B8')
