from src.host_specific_recovery.io.Yaffe_et_al_loader import load_Yaffe_et_al_data
from src.host_specific_recovery.analysis.surrogate_data_analysis import run_surrogate_analysis, run_binomial_test
from src.host_specific_recovery.visualizations.plot_surrogate_data_analysis import plot_SDA, plot_SDA_slow
from src.host_specific_recovery.io.writers import save_sda_results, save_sda_plot_results, save_binomial_test_results

dataset = load_Yaffe_et_al_data()
results_sda = run_surrogate_analysis(dataset, timepoints_val=4)

save_sda_results(outputs=results_sda, base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Yaffe_et_al/surrogate_data_analysis")

results_binom_test = run_binomial_test(results_sda)

plot_SDA(outputs=results_sda, fig_size=(20, 4), sur_color="#8C2928", real_color="#51A246", ymin=-5,
         show_y=False, legend=False, ymax=20, dir=None, y_title='Standardized Jaccard similarity', naive=True)

results_plot = plot_SDA(outputs=results_sda, fig_size=(22, 4), sur_color="#8C2928", real_color="#51A246",
                        ymin=-5, ymax=20, dir=None,
                        y_title='Standardized Jaccard similarity', naive=False)

save_sda_plot_results(results_plot,
                      base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Yaffe_et_al/surrogate_data_analysis")
plot_SDA_slow(outputs=results_sda, significance=results_plot["pvals_labels"], fig_size=(20, 4),
              sur_color="#8C2928", real_color="#51A246", ymin=-5, ymax=20, dir=None)

results_binomial_test = run_binomial_test(results_sda)

save_binomial_test_results(results_binomial_test,
                           base_dir="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                               "Recovery_from_antibiotics/results/Yaffe_et_al/surrogate_data_analysis")
