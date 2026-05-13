from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.species_proportions_heatmap_analysis import (
    run_species_proportios_heatmap_analysis)
from src.host_specific_recovery.visualizations.plot_species_proportions_heatmap_analysis import (
    plot_proportions_heatmap, plot_pie_chart)

dataset = load_Messaoudene_et_al_data()
outputs = run_species_proportios_heatmap_analysis(dataset, tau=2, control=True)
plot_proportions_heatmap(outputs)
plot_pie_chart(outputs, weighted=True)
plot_pie_chart(outputs, weighted=False)
