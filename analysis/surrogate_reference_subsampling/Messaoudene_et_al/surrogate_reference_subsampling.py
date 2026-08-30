from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.host_specific_recovery.analysis.surrogate_reference_subsampling_analysis import (
    run_reference_subsampling_analysis)
from src.host_specific_recovery.io.Messaoudene_et_al_loader import load_Messaoudene_et_al_data
from src.host_specific_recovery.visualizations.plot_surrogate_reference_subsampling_analysis import (
    plot_reference_subsampling_summary)

REFERENCE_SUBSET_SIZES = tuple(range(10, 131, 10))
N_REPEATS = 1000
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "Messaoudene_et_al"
    / "surrogate_reference_subsampling"
    / "reference_subsampling_results.csv"
)
FIGURE_PATH = RESULT_PATH.with_name("reference_subsampling_summary.png")

def run_Messaoudene_reference_subsampling_analysis(
        subset_sizes=REFERENCE_SUBSET_SIZES,
        n_repeats=N_REPEATS,
        timepoints_val=2,
        method="jensenshannon",
        alpha=0.05,
        random_state=0,
        path=RESULT_PATH,
):
    """Run and optionally save Messaoudene reference-size sensitivity results."""
    dataset = load_Messaoudene_et_al_data()
    results = run_reference_subsampling_analysis(
        dataset=dataset,
        subset_sizes=subset_sizes,
        n_repeats=n_repeats,
        timepoints_val=timepoints_val,
        method=method,
        alpha=alpha,
        random_state=random_state,
        strict=False,
        naive=True,
    )

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(path, index=False)

    return results


def plot_Messaoudene_reference_subsampling_summary(
        results,
        path=FIGURE_PATH,
        show_confidence_only=True,
        alpha=0.05,
):
    """Plot and optionally save the Messaoudene subsampling summary."""
    return plot_reference_subsampling_summary(
        results,
        path=path,
        show_confidence_only=show_confidence_only,
        alpha=alpha,
        figsize=(10, 4)
    )


if __name__ == "__main__":
    outputs = run_Messaoudene_reference_subsampling_analysis()
    plot_Messaoudene_reference_subsampling_summary(
        outputs,
        show_confidence_only=True,
    )
    print(f"Saved {len(outputs)} rows to {RESULT_PATH}")
    print(f"Saved summary figure to {FIGURE_PATH}")
