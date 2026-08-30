from src.host_specific_recovery.io.Messaoudene_et_al_loader import (load_Messaoudene_et_al_data,
                                                                    load_Messaoudene_et_al_functional_data)
from src.host_specific_recovery.analysis.functional_data_pipeline_analysis import (run_functional_data_pipeline_prepare,
                                                                                   run_functional_data_pipeline_analysis
                                                                                   )
from src.host_specific_recovery.visualizations.plot_functional_data_pipeline_analysis import (plot_two_hists_with_auc,
                                                                                              plot_similarity_network,
                                                                                              plot_effectsize_vs_effectsize,
                                                                                              plot_functional_redundancy_over_time,
                                                                                              plot_functional_specificity_over_time)
from src.host_specific_recovery.io.writers import (save_functional_data_pipeline_analysis_results,
                                                   save_functional_data_pipeline_prepare_results)

dataset = load_Messaoudene_et_al_data(dir='C:/Users/USER/OneDrive/Desktop/Antibiotics/DAV132/picrust2_data/')
fun_data = load_Messaoudene_et_al_functional_data()
outputs_prep = run_functional_data_pipeline_prepare(dataset['data'], fun_data['rename_map'],
                                                    fun_data['PATH_contrib_path'], fun_data['PATH'], dataset)
g = [f[2] for f in outputs_prep['fun_redundancy_lst']]
outputs_pipeline = run_functional_data_pipeline_analysis(outputs_prep, dataset, subject_idx=4)
path = ("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results/Messaoudene_et_al/"
        "functional_analysis")

save_functional_data_pipeline_prepare_results(outputs_prep, path)
save_functional_data_pipeline_analysis_results(outputs_pipeline, path)

outputs_pipeline_last = run_functional_data_pipeline_analysis(outputs_prep, dataset, subject_idx=4, only_last=True)

base_path = ("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results/"
             "Messaoudene_et_al/functional_analysis")

functional_redundancy_timepoint_labels = ["Baseline", "Day 3 (ABX)", "Day 6 (ABX)", "Day 9", "Day 12",
                                          "Day 16", "Day 25", "Day 37"]
functional_redundancy_figsize = (6, 5)

plot_functional_redundancy_over_time(
    outputs_prep["fun_redundancy_lst"],
    timepoint_labels=functional_redundancy_timepoint_labels,
    figsize=functional_redundancy_figsize,
    path=base_path + "/functional_redundancy_over_time",
)

plot_functional_specificity_over_time(
    outputs_prep["specificity_similarity_lst"],
    timepoint_labels=functional_redundancy_timepoint_labels,
    figsize=functional_redundancy_figsize,
    y_min=0.65,
    path=base_path + "/functional_specificity_over_time",
)

plot_similarity_network(S_AC=outputs_pipeline_last["S_lost_colon"], S_BC=outputs_pipeline_last["S_lost_transient_comb"],
                        S_AD=outputs_pipeline_last["S_base_colon"], S_BD=outputs_pipeline_last["S_base_transient_comb"],
                        node_size=80, threshold_quantile=0.8, gap=0.1, figsize=(15, 25), gap_AB=0.1, gap_CD=0.1,
                        path=base_path + "/no_lost", hide_set="C")
plot_similarity_network(S_AC=outputs_pipeline_last["S_lost_colon"], S_BC=outputs_pipeline_last["S_lost_transient_comb"],
                        S_AD=outputs_pipeline_last["S_base_colon"], S_BD=outputs_pipeline_last["S_base_transient_comb"],
                        node_size=80, threshold_quantile=0.8, gap=0.1, figsize=(15, 25), gap_AB=0.1, gap_CD=0.1,
                        path=base_path + "/no_base", hide_set="D")
plot_similarity_network(S_AC=outputs_pipeline_last["S_lost_colon"], S_BC=outputs_pipeline_last["S_lost_transient_comb"],
                        S_AD=outputs_pipeline_last["S_base_colon"], S_BD=outputs_pipeline_last["S_base_transient_comb"],
                        node_size=80, threshold_quantile=0.8, gap=0.1, figsize=(15, 25), gap_AB=0.1, gap_CD=0.1,
                        path=base_path + "/full")

plot_effectsize_vs_effectsize(outputs_pipeline["AUC_mean_colon_transient_vs_lost_valid"],
                              outputs_pipeline["adjusted_pvalues_mean_colon_transient_vs_lost_valid"],
                              outputs_pipeline["AUC_mean_colon_transient_vs_base_valid"],
                              outputs_pipeline["adjusted_pvalues_mean_colon_transient_vs_base_valid"])
plot_similarity_network(S_AC=outputs_pipeline["S_lost_colon"], S_BC=outputs_pipeline["S_lost_transient_comb"],
                        S_AD=outputs_pipeline["S_base_colon"], S_BD=outputs_pipeline["S_base_transient_comb"],
                        node_size=80, threshold_quantile=0.8, gap=0.1, figsize=(15, 25), gap_AB=0.1, gap_CD=0.1,
                        path=None)
plot_two_hists_with_auc(outputs_pipeline["S_base_colon"].mean(axis=1),
                        outputs_pipeline["S_base_transient_comb"].mean(axis=1),
                        labels=("Colonizers", "Transient"), n_bins=10, alpha=0.7, path=base_path + "/auc_base",
                        x_fontsize=27.5, tick_fontsize=25, legend_fontsize=25, xlabel=" ", ylabel=" ")
plot_two_hists_with_auc(outputs_pipeline["S_lost_colon"].mean(axis=1),
                        outputs_pipeline["S_lost_transient_comb"].mean(axis=1),labels=("Colonizers", "Transient"),
                        n_bins=10, alpha=0.7, path=base_path + "/auc_lost", x_fontsize=27.5, tick_fontsize=25,
                        legend_fontsize=25, xlabel=" ", ylabel=" ")
