from pathlib import Path

import pandas as pd

from src.host_specific_recovery.analysis.new_species_first_detection_analysis import (
    run_new_species_first_detection_analysis,
)
from src.host_specific_recovery.io.Sewunet_et_al_loader import (
    load_Sewunet_et_al_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "Sewunet_et_al"
    / "new_species_first_detection_analysis"
)


def run_Sewunet_new_species_first_detection_analysis() -> dict:
    """Run first-detection counts for all Sewunet participants."""
    dataset = load_Sewunet_et_al_data()
    return run_new_species_first_detection_analysis(dataset)


def outputs_to_table(outputs: dict) -> pd.DataFrame:
    """Convert analysis arrays to a subject-indexed count table."""
    day_columns = [
        f"Day {float(day):g}"
        for day in outputs["times_post_abx"]
    ]
    table = pd.DataFrame(
        outputs["new_species_counts"],
        index=outputs["subject_ids"],
        columns=day_columns,
    )
    table.index.name = "Subject"
    table["Total excluding last day"] = outputs[
        "total_new_species_counts_excluding_last"
    ]
    table["Colonized"] = outputs["colonized_new_species_counts"]
    table["Total"] = outputs["total_new_species_counts"]
    return table


def summarize_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return column means and sample standard deviations across subjects."""
    summary = table.agg(["mean", "std"])
    summary.index = ["Mean", "Standard deviation"]
    summary.index.name = "Statistic"
    return summary


if __name__ == "__main__":
    analysis_outputs = run_Sewunet_new_species_first_detection_analysis()
    results_table = outputs_to_table(analysis_outputs)
    summary_table = summarize_table(results_table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "new_species_first_detection_counts.csv"
    summary_path = RESULTS_DIR / "new_species_first_detection_summary.csv"
    results_table.to_csv(output_path)
    summary_table.to_csv(summary_path)

    print(results_table.to_string())
    print(f"\nSummary across subjects:\n{summary_table.to_string()}")
    print(f"\nSaved results to {output_path}")
    print(f"Saved summary to {summary_path}")
