from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.analysis.species_proportions_heatmap_analysis import (
    run_species_proportios_heatmap_analysis)
from src.host_specific_recovery.visualizations.plot_species_proportions_heatmap_analysis import (
    plot_proportions_heatmap, plot_pie_chart)

dataset = load_Yaffe_et_al_data()

s1=dataset["baseline_minus_63"].astype(bool).sum(axis=1)
s2=dataset["baseline_minus_2"].astype(bool).sum(axis=1)
s3=dataset["baseline_minus_1"].astype(bool).sum(axis=1)
s4=dataset["baseline"].astype(bool).sum(axis=1)
s5=dataset["abx_1"].astype(bool).sum(axis=1)
s6=dataset["abx_2"].astype(bool).sum(axis=1)
s7=dataset["abx_3"].astype(bool).sum(axis=1)
s8=dataset["abx_4"].astype(bool).sum(axis=1)
s9=dataset["abx"].astype(bool).sum(axis=1)
s10=dataset["post_abx_6"].astype(bool).sum(axis=1)
s11=dataset["post_abx_7"].astype(bool).sum(axis=1)
s12=dataset["post_abx_8"].astype(bool).sum(axis=1)
s13=dataset["post_abx_10"].astype(bool).sum(axis=1)
s14=dataset["post_abx_18"].astype(bool).sum(axis=1)
s15=dataset["post_abx_28"].astype(bool).sum(axis=1)
s16=dataset["post_abx"].astype(bool).sum(axis=1)


import numpy as np
A = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16])

print(A.mean(axis=1))

outputs = run_species_proportios_heatmap_analysis(dataset, tau=4)
base_path = ("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results/Yaffe_et_al/"
             "heatmap/")
plot_proportions_heatmap(outputs, path=None)#base_path + "heatmap")
plot_pie_chart(outputs, weighted=True, path=None)#base_path + "pie_weighted")
plot_pie_chart(outputs, weighted=False, path=None)#base_path + "pie")
