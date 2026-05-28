from src.host_specific_recovery.io.Messaoudene_et_al_loader import (load_Messaoudene_et_al_functional_data,
                                                                    load_Messaoudene_et_al_data)
from src.host_specific_recovery.utils.general_utils import run_permanova, plot_pcoa_by_groups
import matplotlib.pyplot as plt
import pandas as pd

dataset = load_Messaoudene_et_al_data()

fun_data = load_Messaoudene_et_al_functional_data()
metadata = fun_data["metadata"]

baseline_columns = list(dataset["baseline_full_df"].columns)

baseline_metadata = metadata[metadata["Name"].isin(baseline_columns)].copy()
baseline_sex_by_name = baseline_metadata.set_index("Name")["host_sex"].str.lower()
baseline_sex_by_name = (baseline_sex_by_name.reset_index().drop_duplicates().set_index("Name").squeeze())
baseline_bmi_by_name = baseline_metadata.set_index("Name")["host_body_mass_index"]
baseline_bmi_by_name = (baseline_bmi_by_name.reset_index().drop_duplicates().set_index("Name").squeeze())

baseline_bmi_by_name = pd.to_numeric(baseline_bmi_by_name, errors="coerce").apply(
    lambda bmi: "high" if bmi >= 25 else "low"
)
baseline_age_by_name = baseline_metadata.set_index("Name")["Host_age"]
baseline_age_by_name = (baseline_age_by_name.reset_index().drop_duplicates().set_index("Name").squeeze())

baseline_age_by_name = pd.to_numeric(baseline_age_by_name, errors="coerce")
baseline_age_mean = baseline_age_by_name.mean()
baseline_age_by_name = baseline_age_by_name.apply(
    lambda age: "old" if age >= baseline_age_mean else "young"
)

baseline_df = dataset["baseline_full_df"]

permanova_results_sex = run_permanova(baseline_df, baseline_sex_by_name, metric='jensenshannon')
pcoa_results_sex = plot_pcoa_by_groups(baseline_df, baseline_sex_by_name, legend_loc='upper right',
                                       path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                                        "Recovery_from_antibiotics/results/Messaoudene_et_al/"
                                        "baseline_metadata_analysis/sex_analysis")
plt.show()

permanova_results_bmi = run_permanova(baseline_df, baseline_bmi_by_name, metric='jensenshannon')
pcoa_results_bmi = plot_pcoa_by_groups(baseline_df, baseline_bmi_by_name, legend_loc='upper right',
                                       path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                                        "Recovery_from_antibiotics/results/Messaoudene_et_al/"
                                        "baseline_metadata_analysis/bmi_analysis")
plt.show()

permanova_results_age = run_permanova(baseline_df, baseline_age_by_name, metric='jensenshannon')
pcoa_results_age = plot_pcoa_by_groups(baseline_df, baseline_age_by_name, legend_loc='upper right',
                                       path="C:/Users/USER/OneDrive/Desktop/Recovery_from_antibiotics/"
                                        "Recovery_from_antibiotics/results/Messaoudene_et_al/"
                                        "baseline_metadata_analysis/age_analysis")
plt.show()

print(0)
