from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.analysis.surrogate_data_analysis import run_surrogate_analysis, run_binomial_test
from src.host_specific_recovery.visualizations.plot_surrogate_data_analysis import plot_SDA, plot_SDA_slow
from src.host_specific_recovery.io.writers import save_sda_results, save_sda_plot_results, save_binomial_test_results

dataset = load_Messaoudene_et_al_data()
results_sda_rjsd = run_surrogate_analysis(dataset, timepoints_val=2, method="jensenshannon")

plot_SDA(outputs=results_sda_rjsd, fig_size=(9, 4), sur_color="#8C2928", real_color="#51A246", ymin=0.1,
         ymax=0.8, dir=None, y_title='rJSD similarity', naive=True, show_y=True,
         std=False)#, #base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                    #           "Recovery_from_antibiotics/results/Messaoudene_et_al/surrogate_data_analysis")

results_sda = run_surrogate_analysis(dataset, timepoints_val=2)

save_sda_results(outputs=results_sda, base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Messaoudene_et_al/surrogate_data_analysis")

results_binom_test = run_binomial_test(results_sda)

plot_SDA(outputs=results_sda, fig_size=(9, 4), sur_color="#8C2928", real_color="#51A246", ymin=-4.9314556423269575,
         ymax=10.899442828117659, dir=None, y_title='Standardized Jaccard similarity', naive=True)


results_plot = plot_SDA(outputs=results_sda, fig_size=(9.25, 4), sur_color="#8C2928",
                                                 real_color="#51A246", show_y=False, legend=False,
                                                 ymin=-4.9314556423269575, ymax=10.899442828117659, dir=None,
                                                 y_title='Standardized Jaccard similarity', naive=False)

save_sda_plot_results(results_plot,
                      base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Messaoudene_et_al/surrogate_data_analysis")

plot_SDA_slow(outputs=results_sda, success_idx_input=results_plot["success_idx"],
              fail_idx_input=results_plot["fail_idx"], fig_size=(9.25, 4),
              sur_color="#8C2928", real_color="#51A246", show_y=False, legend=False, ymin=-4.082636696859447,
              ymax=8.431722910417848, dir=None)

results_binomial_test = run_binomial_test(results_sda)

save_binomial_test_results(results_binomial_test,
                           base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Messaoudene_et_al/surrogate_data_analysis")
