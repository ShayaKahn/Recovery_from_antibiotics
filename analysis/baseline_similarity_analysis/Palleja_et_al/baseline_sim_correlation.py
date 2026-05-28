import numpy as np
from pathlib import Path

from src.host_specific_recovery.io.Palleja_et_al_loader import load_Palleja_et_al_data
from src.host_specific_recovery.utils.general_utils import plot_pcoa_and_distance_heatmap, plot_correlation
from src.host_specific_recovery.analysis.surrogate_data_analysis import run_surrogate_analysis
from src.host_specific_recovery.visualizations.plot_surrogate_data_analysis import plot_SDA

dataset = load_Palleja_et_al_data()
results_sda = run_surrogate_analysis(dataset, timepoints_val=1, method="jensenshannon")
plot_results = plot_SDA(outputs=results_sda, fig_size=(9, 4), sur_color="#8C2928", real_color="#51A246",
                        ymin=-4.9314556423269575, ymax=10.899442828117659, dir=None,
                        y_title='Standardized rJSD similarity', naive=True)

order = np.array(plot_results["order"])

labels = np.array(dataset["filtered_keys"])[order]

df_baseline = dataset["baseline_df"]

baseline_rjsd = plot_pcoa_and_distance_heatmap(
    df_baseline,
    metric="rjsd",
    title_prefix="Baseline"
)

baseline_rjsd_dist = baseline_rjsd['distance_matrix']
baseline_rjsd_dist_no_diag = baseline_rjsd_dist.to_numpy(dtype=float).copy()
np.fill_diagonal(baseline_rjsd_dist_no_diag, np.nan)
baseline_rjsd_mean_dist = np.nanmean(baseline_rjsd_dist_no_diag, axis=0)[order]

obs_z_sorted = np.asarray(plot_results['obs_z_sorted'], dtype=float)
mean_dist_sorted = np.asarray(baseline_rjsd_mean_dist, dtype=float)
valid = np.isfinite(mean_dist_sorted) & np.isfinite(obs_z_sorted)

x = mean_dist_sorted[valid]
y = obs_z_sorted[valid]
names = labels[valid]

correlation_plot = plot_correlation(
    x=x,
    y=y,
    labels=names,
    x_title="Mean rJSD similarity to other baselines",
    y_title="SDA: Standardized rJSD similarity",
    x_annotation=0.35, y_annotation=0.2
)

output_dir = Path("C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/Recovery_from_antibiotics/results"
                  "/Palleja_et_al/baseline_similarity_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
correlation_plot["figure"].savefig(
    output_dir / "baseline_rjsd_sda_correlation.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)
