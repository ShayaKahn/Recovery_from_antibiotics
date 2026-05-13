from src.host_specific_recovery.io.Sewunet_et_al_loader import load_Sewunet_et_al_data
from src.host_specific_recovery.analysis.assembly_times_analysis import calculate_characteristic_time
from src.host_specific_recovery.visualizations.plot_assembly_times_analysis import plot_characteristic_time

dataset = load_Sewunet_et_al_data()
outputs = calculate_characteristic_time(dataset)
plot_characteristic_time(outputs, '#B82E2E', 'o', 'Sewunet et al. 2024',2,
                         None, None, save_fig=False)
