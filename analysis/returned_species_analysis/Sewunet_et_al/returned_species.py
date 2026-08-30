import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.host_specific_recovery.analysis.returned_species_analysis import (
    ReturnedSpeciesAnalysis,
)
from src.host_specific_recovery.io.Sewunet_et_al_loader import (
    load_Sewunet_et_al_data,
)
from src.host_specific_recovery.visualizations.plot_returned_species_analysis import (
    plot_returned_species_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "Sewunet_et_al"
    / "returned_species_analysis"
)


def run_Sewunet_returned_species_analysis(
        subject_id,
        non_rarefied=True,
        survived_reduction_threshold=10.0,
):
    """Find taxa that first return at the final follow-up for one subject."""
    dataset = load_Sewunet_et_al_data()
    return ReturnedSpeciesAnalysis(
        dataset,
        non_rarefied=non_rarefied,
        survived_reduction_threshold=survived_reduction_threshold,
    ).run(subject_id)


def run_Sewunet_non_rarefied_returned_species_analysis(
        subject_id,
        survived_reduction_threshold=10.0,
):
    """Run the returned-species analysis with non-rarefied abundances."""
    return run_Sewunet_returned_species_analysis(
        subject_id,
        non_rarefied=True,
        survived_reduction_threshold=survived_reduction_threshold,
    )


def plot_Sewunet_returned_species(
        outputs,
        log_scale=True,
        show_legend=True,
        show_low_count_annotation=False,
        top_x=3,
        path=None,
):
    """Plot returned-taxon abundance trajectories for one Sewunet subject."""
    return plot_returned_species_analysis(
        outputs,
        log_scale=log_scale,
        path=path,
        figsize=(14, 10),
        line_width=2.5,
        marker_size=16,
        marker_alpha=0.7,
        abx_region_width=5.0,
        timepoint_spread=0.64,
        xticks_fontsize=20,
        ylabel_fontsize=32,
        ytick_fontsize=20,
        legend_fontsize=11,
        show_legend=show_legend,
        show_low_count_annotation=show_low_count_annotation,
        top_x=top_x,
        legend_loc="upper left",
        legend_bbox_to_anchor=(0.4, 0.53),
        tick_length=10,
        tick_width=2.5,
    )

def _safe_filename(value) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot taxa first detected at a subject's final post-ABX sample."
    )
    parser.add_argument(
        "subject_id",
        nargs="?",
        help="Subject ID, for example '2 AMC'. Omit it to analyze all subjects.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use a linear rather than logarithmic y-axis.",
    )
    parser.add_argument(
        "--non-rarefied",
        "--non-rarified",
        dest="non_rarefied",
        action="store_true",
        default=True,
        help="Use non-rarefied relative abundances and sample-specific sequencing depths.",
    )
    parser.add_argument(
        "--rarefied",
        dest="non_rarefied",
        action="store_false",
        help="Use rarefied relative abundances and the common sequencing depth N_0.",
    )
    parser.add_argument(
        "--survived-reduction-threshold",
        type=float,
        default=10.0,
        metavar="FOLD",
        help=(
            "Minimum baseline-to-ABX fold reduction for survived species "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--top-x",
        type=int,
        default=7,
        metavar="X",
        help="Plot at most the top X taxa in each category (default: 7).",
    )
    parser.add_argument(
        "--no-legend",
        dest="show_legend",
        action="store_false",
        default=False,
        help="Hide the figure legend (shown by default).",
    )
    parser.add_argument(
        "--show-low-count-annotation",
        dest="show_low_count_annotation",
        action="store_true",
        default=False,
        help="Show the sampling-sensitive low-count-region text and arrow.",
    )
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    arguments = parser.parse_args()

    if (
            not math.isfinite(arguments.survived_reduction_threshold)
            or arguments.survived_reduction_threshold <= 0
    ):
        parser.error(
            "--survived-reduction-threshold must be a positive finite number."
        )
    if arguments.top_x <= 0:
        parser.error("--top-x must be a positive integer.")

    if arguments.output is not None and arguments.subject_id is None:
        parser.error("--output can only be used when a single subject_id is supplied.")

    dataset = load_Sewunet_et_al_data()
    analysis = ReturnedSpeciesAnalysis(
        dataset,
        non_rarefied=arguments.non_rarefied,
        survived_reduction_threshold=arguments.survived_reduction_threshold,
    )
    subject_ids = (
        [arguments.subject_id]
        if arguments.subject_id is not None
        else analysis.subject_ids
    )

    total_returned_taxa = 0
    total_first_returned_taxa = 0
    total_reduced_survived_taxa = 0
    subjects_with_returned_taxa = 0
    subjects_with_first_returned_taxa = 0
    subjects_with_reduced_survived_taxa = 0
    for subject_id in subject_ids:
        outputs = analysis.run(subject_id)
        returned_taxa = outputs["n_returned_species"]
        first_returned_taxa = outputs["n_first_returned_species"]
        reduced_survived_taxa = outputs["n_reduced_survived_species"]
        total_returned_taxa += returned_taxa
        total_first_returned_taxa += first_returned_taxa
        total_reduced_survived_taxa += reduced_survived_taxa

        print(
            f"\nSubject {subject_id}: {first_returned_taxa} taxa returned at the first "
            f"and persisted through all post-ABX samples; {returned_taxa} taxa returned "
            f"at the last timepoint; "
            f"{reduced_survived_taxa} survived taxa with "
            f">{arguments.survived_reduction_threshold:g}-fold ABX reduction"
        )
        if not returned_taxa and not first_returned_taxa and not reduced_survived_taxa:
            print("No figure created because this subject has no qualifying taxa.")
            continue

        if first_returned_taxa:
            subjects_with_first_returned_taxa += 1
            print("Taxa returned at the first and detected in all post-ABX samples:")
            print(outputs["first_returned_species"].to_string(index=False))
        if returned_taxa:
            subjects_with_returned_taxa += 1
            print("Taxa returned at the last post-ABX timepoint:")
            print(outputs["returned_species"].to_string(index=False))
        if reduced_survived_taxa:
            subjects_with_reduced_survived_taxa += 1
            print("Reduced survived taxa:")
            print(outputs["reduced_survived_species"].to_string(index=False))
        output_path = arguments.output or (
            FIGURE_DIR
            / (
                f"subject_{_safe_filename(subject_id)}_returned_species"
                f"{'_non_rarefied' if arguments.non_rarefied else ''}.png"
            )
        )
        figure, _ = plot_Sewunet_returned_species(
            outputs,
            log_scale=not arguments.linear,
            show_legend=arguments.show_legend,
            show_low_count_annotation=arguments.show_low_count_annotation,
            top_x=arguments.top_x,
            path=output_path,
        )
        plt.close(figure)
        print(f"Saved figure to {output_path}")

    if arguments.subject_id is None:
        print(
            f"\nFinished {len(subject_ids)} subjects: "
            f"{subjects_with_first_returned_taxa} had {total_first_returned_taxa} taxa "
            f"returning at the first and persisting through all post-ABX samples; "
            f"{subjects_with_returned_taxa} had {total_returned_taxa} taxa returning "
            f"at the last post-ABX timepoint; "
            f"{subjects_with_reduced_survived_taxa} had "
            f"{total_reduced_survived_taxa} survived taxa with "
            f">{arguments.survived_reduction_threshold:g}-fold ABX reduction."
        )
