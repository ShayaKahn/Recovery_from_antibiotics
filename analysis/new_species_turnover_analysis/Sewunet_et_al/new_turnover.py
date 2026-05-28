from src.host_specific_recovery.io.Sewunet_et_al_loader import load_Sewunet_et_al_data
from src.host_specific_recovery.analysis.new_species_turnover_analysis import run_new_species_turnover
from src.host_specific_recovery.visualizations.plot_new_species_turnover_analysis import  connect_columns_plot

dataset = load_Sewunet_et_al_data()
outputs = run_new_species_turnover(dataset)
ax = connect_columns_plot(outputs,
                          path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/"
                               "results/Sewunet_et_al/new_species_turnover/new_species_turnover_plot")