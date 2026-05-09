from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.null_model_analysis import run_null_model_analysis
from src.host_specific_recovery.visualizations.plot_null_model_analysis import plot_NM_violin
import numpy as np

dataset = load_Messaoudene_et_al_data()
results_nm = run_null_model_analysis(dataset, timepoints_val=1)
order = np.load("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Messaoudene_et_al/surrogate_data_analysis/order.npy")

fig, ax = plot_NM_violin(outputs=results_nm, fig_size=(9.25, 4), sur_color="#787878", real_color="#1A72B780",
                         custom_order=order, show_y=False, legend=False, ymin=None, ymax=None, dir=None)
fig.show()
