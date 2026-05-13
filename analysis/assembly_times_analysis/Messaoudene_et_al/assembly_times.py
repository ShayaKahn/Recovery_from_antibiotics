from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.assembly_times_analysis import calculate_characteristic_time
from src.host_specific_recovery.visualizations.plot_assembly_times_analysis import plot_characteristic_time

dataset = load_Messaoudene_et_al_data()
outputs = calculate_characteristic_time(dataset)
plot_characteristic_time(outputs, '#1A71B8', 's', 'Messaoudene et al. 2024',2,
                         None, None, save_fig=False)
