from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.analysis.assembly_times_analysis import calculate_characteristic_time
from src.host_specific_recovery.visualizations.plot_assembly_times_analysis import plot_characteristic_time

dataset = load_Yaffe_et_al_data()
outputs = calculate_characteristic_time(dataset)
plot_characteristic_time(outputs, '#B82E2E', 'o', 'Yaffe et al. 2025',2,
                         "C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/"
                         "results/Yaffe_et_al/assembly_times/", "characteristic_time_plot", save_fig=True)
