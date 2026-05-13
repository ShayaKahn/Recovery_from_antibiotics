from src.host_specific_recovery.io.Palleja_et_al_loader import load_Palleja_et_al_data
from src.host_specific_recovery.analysis.assembly_times_analysis import calculate_characteristic_time
from src.host_specific_recovery.visualizations.plot_assembly_times_analysis import plot_characteristic_time

dataset = load_Palleja_et_al_data()
outputs = calculate_characteristic_time(dataset)
plot_characteristic_time(outputs, '#F58518', '^', 'Palleja et al. 2018',5,
                        None, None, save_fig=False)
